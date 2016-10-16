---
layout: post
title: How to continuously update the Elasticsearch index
description: "A step by step guide on how to update the Elasticsearch index when new posts come in"
modified: 2016-15-15
tags: [python, elasticsearch, hacker news]
image:
  feature: abstract-7.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

We saw [earlier]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}) how to create an index containing the posts from the Hacker News _who's hiring_ thread.

Since during the course of the month new posts (thus new jobs) are added, we want to update the script so that it will add only the new posting without overwriting the ones that are already there.

The first step is again to create query the [Hacker News API](https://github.com/HackerNews/API), to see what posts are currently online, just as before.

{% highlight py %}
thread = 12627852

currents_posts = fetch_hn_data(thread)['kids']
{% endhighlight %}

For the definition of `fetch_hn_data` pleae refer to the [previous post]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}) or the corresponding [GitHub repo](https://github.com/hdmetor/HNCrawler). The function is doing what you would expect, with some text cleaning on top.

Now we need to figure out what items are already in our index. The first thing that comes to mind, is to do a query that will return _every_ element of the index, i.e.

{% highlight python %}
{
    'query': {
        "match_all" : {}
    }
}
{% endhighlight %}

While this would technically work (and for now there would be no difference since we have only indexed the posts from one month), we want to do better.

Each child post has a `parent` filed, so we can just impose the extra condition in the query

{% highlight python %}
{
    'query': {
        'term': {"parent": thread}
    }
}
{% endhighlight %}


Instead of querying the index via the usual `search` method, we want to use the [`scroll` API](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-request-scroll.html), which handles bigger requests better. The [python bindings](http://elasticsearch-py.readthedocs.io/en/master/helpers.html) are really easy to use.

{% highlight python %}
query = {
    'query': {
        'term': {"parent": thread}
    }
}

older_posts_gen = helpers.scan(es, query)
{% endhighlight %}

As for now, `older_posts_gen` is a generator where each item is a `dict` that contains the metadata and the full data for each the posts. Since we only need the post id (stored as the `_id` metadata value), we can suppress the full data in the response (corresponding to the `_source` value in the metadata).

{% highlight python %}
query = {
    "_source": False,
    'query': {
        'term': {"parent": thread}
    }
}

older_posts_gen = helpers.scan(es, query)
{% endhighlight %}

Let's now grab only the ids.

{% highlight python %}
old_posts_ids = {int(item['_id']) for item in older_posts_gen}
{% endhighlight %}

We can now find which are the new posts, and index only those

{% highlight python %}
new_posts_ids = set(currents_posts) - old_posts_ids
if new_posts_ids:
    print("There are {} new posts!".format(len(new_posts_ids)))
    actions = [format_data_for_action(r) for r in new_posts_ids]
    list(parallel_bulk(es, actions))
{% endhighlight %}


Note that using a the `parallel_bulk` here is an overkill, but we want to maximize code reuse.

Let's put this code in function. Given a thread id, the function will index all the posts that are new for the index, and not deleted from the website.


{% highlight python %}
def update_thread(thread, es=None):
    """Put into the index new child posts"""

    if not es:
        es = Elasticsearch()
    currents_posts = fetch_hn_data(thread)['kids']
    query = {
        "_source": False,
        'query': {
            'term': {"parent": thread}
        }
    }
    older_posts_gen = helpers.scan(es, query)
    old_posts_ids = {int(item['_id']) for item in older_posts_gen}
    new_posts_ids = set(currents_posts) - old_posts_ids
    if new_posts_ids:
        print("There are {} new posts!".format(len(new_posts_ids)))
        actions = [format_data_for_action(r) for r in new_posts_ids if format_data_for_action(r)]
        list(parallel_bulk(es, actions))

{% endhighlight %}

# What month is it?
![Month]({{site.url}}/assets/2016/month.jpg)


There is one last piece of information we want to retreive: the thread id of the latest who's hiring thread. Luckly such postings are done by the same user (`whoshiring`), and the API let's us perform a query per user.

From the [bot activity](https://news.ycombinator.com/submitted?id=whoishiring) you can see that we need to retreive the latest 3 posts, and find the correct one among those.

First of all, we need a more generic version of the fetch data function.

{% highlight python %}
def fetch_hn_data(thread, type_='item'):
    url = "https://hacker-news.firebaseio.com/v0/{}/{}.json".format(type_, thread)
    r = requests.get(url)
    assert r.status_code == 200
    data = r.json()
    if type == 'item':
        data = clean_text(data)
    return data
{% endhighlight %}

Then we can find the correct thread:

{% highlight py %}
def find_hiring_thread():
    all_posts = fetch_hn_data('whoishiring', type_='user')['submitted']
    last_three_posts = sorted(all_posts)[-3:]
    for post in last_three_posts:
        data = fetch_hn_data(post)
        if 'is hiring?' in fetch_hn_data(post)['title']:
            break
    return post

{% endhighlight %}

We can just complete the scirpt with

{% highlight python %}
if __name__ == '__main__':
    es = Elasticsearch()
    assert es.ping(), "Elasticsearch not started properly"
    if not es.indices.exists(index='hn'):
        print('The index does not extis, creating one')
        es.indices.create(index='hn', body=mappings)
    print('Looking for the latests hiring thread...')
    thread = find_hiring_thread()
    print('updating index...')
{% endhighlight %}


And make sure it run one the computer is started. We won't give explicit instrucions about it, because this is system depepndet, and there are plenty of resources onoine for that. Also if you never turn off your computer you might want to set up a cron job to run this reguraly.

Please remember that the code is available in [this](https://github.com/hdmetor/HNCrawler) GitHub repo.
