---
layout: post
title: "How I found a job in San Francisco with Hackernews and Elasticsearch"
date: 2016-10-07
---

Recently I have decided to move to San Francisco, and I had only one problem: finding a job.

I knew two things:

- I didn't want to apply to one of the big companies

- I wanted a Machine Learning job

After doing some obvious Google searches, I decided to find a better way, especially considering that anonymous online applicants are rarely considered. (to my surprise, I obtained an onsite interview with a company I applied online.)

What I wanted was a list of companies offering machine learning positions, and possibly a smoother application process.

I decided to use the monthly Hacker News [who's hiring](https://www.google.com/search?q=hacker+news+who%27s+hiring) to help me with the search. Usually you can contact somebody who is already working in the team (and not a recruiter), so it's a better introduction.

At the time of writing, the last of such thread is located [here](https://news.ycombinator.com/item?id=12627852).

Since this thread is now very popular, there are way too many postings to look them one by one, so I needed to filter our the uninteresting ones. What I wanted to _keep_ was

- a top level thread. This is where usually jobs are posted. Replies are mostly used to ask for additional info

- a job located in SF or in the Bay Area

- a job related to my area of interest

Furthermore, some jobs are added in during the course of the month, so I needed a way of keeping track of what was already scraped and what was new.

Finally I wanted an easy way to read trough the jobs, so that I could pick the ones were interesting to me, and move from there.

# First of all: the data

Of course the first thing we need is to get the data from the thread. Hacker News conveniently offers an [API](https://github.com/HackerNews/API), that gives all the first level posts (called `kids` in the response) of a given story. This is exactly what we need.

{% highlight py %}
import requests
from bs4 import BeautifulSoup

thread = 12627852

def clean_text(data):
    try:
        clean_text = BeautifulSoup(data['text'], "html.parser").text
        data['text'] = clean_text
        return data
    except KeyError:
          return {}

def fetch_hn_data(thread):
    url = "https://hacker-news.firebaseio.com/v0/item/{}.json".format(thread)
    r = requests.get(url)
    assert r.status_code == 200
    data = r.json()
    data = clean_text(data)
    return data
data = fetch_hn_data(thread)
{% endhighlight %}

The `text` value from the API response is still in html, which means there is a lot of unnecessary noise. We used BeautifulSoup to obtain the text. We lose paragraph / new line dividers but this doesn't seem to be a problem for now.
The `text` field is not always present. For example the post might be deleted. If that's the case, we return an empty `dict`, for reasons that will be clear in a moment.

In order to be able to search for the right posting (therefore job) we are going to feed all such posts to [Elasticsearch](https://www.elastic.co/), a fully fledged text search engine.
All the filtering and querying will happen from via the python Elasticsearch package.


# Putting the data into Elasticsearch

Before indexing the documents, we need to create the index with mappings (basically a schema). Elasticsearch in general is smart enough to figure out what type of contant a fields has, but we need to define mappings if we want some sort of special treatment for it. In our case we want the `data['time']` to be recognized as a (unix) timestamp.


{% highlight python %}
from elasticsearch import Elasticsearch
# client init
es = Elasticsearch()

mappings = {
    "mappings" : {
        "post" : {
            "properties" : {
                "time" : {
                    "type" : "date",
                    "format" : "strict_date_optional_time||epoch_millis"
                }
            }
        }
    }
}
es.indices.create(index='hn', body=mappings)
{% endhighlight %}

Note that we created an index called `hn` and we claimed (in the `mappings` dict) that such an index will have a document type called `post`.

Let's add our documents now. Note that we are not including the top level thread, because it does not contain jobs.

{% highlight python %}
from elasticsearch.helpers import parallel_bulk

def format_data_for_action(post_id):
    return {
    '_index': 'hn',
    '_type': 'post',
    '_id': post_id,
    '_source': fetch_hn_data(post_id)
}

actions = [format_data_for_action(r) for r in data['kids']]
list(parallel_bulk(es, actions))
{% endhighlight %}

Since we have a lot of posting we are using the `parallel_bulk` helper. The bottleneck is the fetching of the data, because we are sending one request at a time. This is the only case where crawling the web page gives an advantage over the api: all the data are already present in the first response. On the other hand it require way more human time (to figure out how to successfully extract the right elements), so it's ok to have the machine to wait for us. Maybe one day we could update the fetching functionlaity sing the new Python `async` and `await` functionalities (keep in mind that this will put more stress on the server).

If everything worked properly, the index will contain all the published posting and we can start querying it.

# Querying the index

Let's have a sanity check, to make sure we are not off track. Let's search for San Francisco jobs. I'm not sure how many we should find, but I am expecting more than 10

{% highlight python %}
query = {
    "query" : {
        "bool" : {
            "must" : {
                "match": { 'text' : 'san francisco'}
            }
        }
    }}

response = es.search(index='hn', body=query)
response['hits']['total']
>>> 180
{% endhighlight %}

And it worked! This seems a very reasonable result. Note that your mileage may vary, because posts get deleted all / added the time.

So this matches _any_ job with a location in San Francisco. What if we want a San Francisco based job, that contains the 'machine learning' keyword?

We just need to add another clause to the `match` query:

{% highlight python %}
query = {"query" : {
        "bool" : {
            "must" : [{
                "match": {'text' : 'san francisco'}},
                {"match": {'text' : 'machine learning'}}
            ]

        }

    }}

es.search(index='hn', body=query)['hits']['total']
>>> 40
{% endhighlight %}

That's great! Now we have a list of potentially interesting jobs that we can look at.

# What's next?

While this simple method works, it is still too manual to be really useful. Foe example the query above is way too restrictive. In general we want something like ('san franisco' OR 'bay area') AND ('machine learning' OR 'data analysis'). Those elasticsearch json-based queries can be really nested and it's easy to get lost in them.

 In the next posts we are going to address the following issues:

- easily combining elasticsearch queries

- updating the index (without reindexing everything from scratch every time)

- being notified (via email, slack or similar) when an new post become available

- presenting the data in a human friendly way

The code used in this post for this can be found [here](https://github.com/hdmetor/HNCrawler).
