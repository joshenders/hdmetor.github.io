---
layout: post
title: How to combine boolean queries in Elasticsearch
description: 'How to combine elasticsearch queries.'
modified: 2016-10-10
tags: [python, elasticsearch, boolean queries]
image:
  feature: abstract-10.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

In a [previous post]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}) we saw how to use Elasticsearch to search for our dream job among the ones posted on hacker news.

Since Elasticsearch queries are basically JSON it's really easy to lose track when we start nesting them.

In this post we are going to define a Python class that will create the required query (read: JSON) on demand.

# Anatomy of an Elasticsearch query

A query in Elasticsearch in its general form is:

{% highlight python %}
{
   'query': {
        'bool': {
            'must': [ ... ]
            'should': [ ... ]
            'must_not': [ ... ]
        }
    }
}

{% endhighlight %}

Where `must`, `should` and `must_not` correspond to the `AND`, `OR` and `AND NOT` operators respectively.

To use an [earlier example]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}), the following query will require both 'san francisco' and 'machine learning' to be present in the `text` field.

{% highlight python %}
{
    'query': {
        'bool': {
            'must': [
                {'match': {'text': 'san francisco'}},
                {'match': {'text': 'machine learning'}}
            ]
        }
    }
}
{% endhighlight %}

So far so good. But what if we want to nest a boolean query inside another boolean query?

![Deeper]({{site.url}}/assets/2016/deeper.jpg)

For example, what if we want to find something that matches ('san franisco' OR 'bay area') AND ('machine learning' OR 'data analysis')?

The resulting query will be something like this:

{% highlight python %}
{
    'query': {
        'bool': {
            'must': [
              {
                'bool': {
                    'should': [
                        {'match': {'text': 'san francisco'}},
                        {'match': {'text': 'bay area'}}
                    ]
                },
              },
              {
                'bool': {
                    'should': [
                        {'match': {'text': 'machine learning'}},
                        {'match': {'text': 'data analysis'}}
                    ]
                }
              }
            ]
        }
    }
}
{% endhighlight %}

You can see how it's easy to get lost here.

# Python to the rescue!

We want to represent the queries in a more human friendly way.

Before start coding, let's design the class. This will save us so much time afterwards. What are the requirements, i.e. what is the expected behavior of our class (and its methods)?

- it should have arguments that correspond to the three boolean operators

- each operator can accept one or more term

- we need to be able to specify which field we want to search in

- each term can be another query

Let's write down some obvious use cases.

{% highlight python %}
# easy case: one field, one term per field
query = Query(must={'text': 'san francisco'}, should_not={'text': 'new york'})

# one field, more terms
query_2 = Query(should={'text': ['san francisco', 'bay area']})

# multiple fields, multiple terms
query_3 = Query(must={'text': ['san francisco', 'bay area'], 'title': 'hiring'})

# query in a query
inner = query_2
query_outer = Query(must={'query': inner, 'title': 'hiring'})

{% endhighlight %}

Most important of all, our class needs to have a `to_elasticsearch` method to produce the desired json on demand.

Let's start coding something that would work for the first query, and then let's improve on that.

Since we are in the easy case, we can assume that the value passed are `dict`s.
In order to build the ES query, we then need to figure out which arguments have been passed (i.e. are not `None`) and put them in the query.

{% highlight python %}
class Query:

    def __init__(self, must=None, should=None, should_not=None):
        self.must = must
        self.should = should
        self.should_not = should_not

    def to_elasticsearch(self):
        names = ['must', 'should', 'should_not']
        values = [self.must, self.should, self.should_not]
        query = {name: value for name, value in zip(names, values) if value}
        query =  {
            'query': {
                'bool': query
            }
        }

    return query
query = Query(must={'text': 'san francisco'}, should_not={'text': 'new york'})
query.to_elasticsearch()
{% endhighlight %}

So far so good. But what if the the same field has multiple values, as in `query_2`?
Our function needs to adapt. For one, we need to use the same key for all the values. So in the example above the key `text` has to be applied to both `'san francisco'` and `'bay area'`.

{% highlight python %}
def to_elasticsearch(self):
    query = {}
    names = ['must', 'should', 'should_not']
    values = [self.must, self.should, self.should_not]
    for name, value in zip(names, values):
        if not value:
            continue
        for field_name, field_values in value.items():
            # field_name = 'text', field_values = ['san francisco', 'bay area']
            query[name] = [{'match': {field_name: v}} for v in field_values]

        query =  {
            'query': {
                'bool': query
            }
        }
    return query
Query.to_elasticsearch = to_elasticsearch
query_2 = Query(should={'text': ['san francisco', 'bay area']})
query_2.to_elasticsearch()
{% endhighlight %}


Now we got it working for lists. What happens if we try mixed case like

{% highlight python %}
query_3 = Query(must={'text': ['san francisco', 'bay area'], 'title': 'hiring'})
{% endhighlight %}

There are 2 things that will break:

- `query[name] = ...` will overwrite previous results (in our case the `title` field will overwirte the `text` one)

- `for v in field_values` part would not behave as expected (e.g. it will unpack a string)

To fix the first problem, we can just make `query` a `defaultdict` and `extend` it. To avoid problems communicating with the Elasticsearch client, we will convert back to a regular `dict` before returning.

{% highlight python %}
from collections import defaultdict

def to_elasticsearch(self):
    query = defaultdict(list)
    names = ['must', 'should', 'should_not']
    values = [self.must, self.should, self.should_not]
    for name, values in zip(names, values):
        if not values:
            continue
        for field_name, field_values in values.items():
            # field_name = 'text', field_values = ['san francisco', 'bay area']
            query[name].extend([{'match': {field_name: v}} for v in field_values])

        query =  {
            'query': {
                'bool': dict(query)
            }
        }
    return query

Query.to_elasticsearch = to_elasticsearch
query_3 = Query(must={'text': ['san francisco', 'bay area'], 'title': 'hiring'})
query_3.to_elasticsearch()
{% endhighlight %}

The most elegant way of solving the second problem, is to transform the input into a standard way. In that way our `to_elasticsearch` method will be independent of the original input form.

For each (non null) argument, we want to make sure that its values are wrapped in a list.

{% highlight python %}
def __init__(self, must=None, should=None, should_not=None):
    self.must = self.preprocess(must)
    self.should = self.preprocess(should)
    self.should_not = self.preprocess(should_not)

def preprocess(self, field):
    if not field:
        return None
    for key, value in field.items():
        if not isinstance(value, list):
            field[key] = [value]
    return field

Query.__init__ = __init__
Query.preprocess = preprocess

query_3 = Query(must={'text': ['san francisco', 'bay area'], 'title': 'hiring'})
query_3.to_elasticsearch()
{% endhighlight %}

Or, in a more compact way:

{% highlight python %}
def preprocess(self, field):
    return {k: v if isinstance(v, list) \
        else [v] for k,v in field.items()} of field \
        else None
{% endhighlight %}

Some stuff to note here:

- if the `field` is `None`, we want to keep that way, not wrap it into a list. Returning `{}` would be fine too

- `[value]` is not the same as `list(value)`

- `query_1` output is now changed (there is an extra list), but it's still a valid ES query, and it returns the same result.


# It's turtles all the way down

Now the interesting part: we want to combine a [query withing a query](http://theinceptionbutton.com/).

Let's do this step by step. First of all, we want the method `to_elasticsearch` to expand any inner queries, if present. Since inner queries are still instsances of `Query` we can call the same method on them.
Therefore, we need to distinguish between a real `Query` object and just a query term.

The minimal edit to the previous code will be something like this:

{% highlight python %}
def to_elasticsearch(self):
    query = defaultdict(list)
    names = ['must', 'should', 'should_not']
    values = [self.must, self.should, self.should_not]
    for name, values in zip(names, values):
        if not values:
            continue
        for field_name, field_values in values.items():
            # field_name = 'text', field_values = ['san francisco', 'bay area']
            # OR
            # field_name = 'query',  field_values = some isntance of Query
            query[name].extend([
                    v.to_elasticsearch() if isinstance(v, Query)
                    else {'match': {field_name: v}}
                    for v in field_values])

        query =  {
            'query': {
                'bool': dict(query)
            }
        }
    return query
{% endhighlight %}

Just to reiterate: everything is the same as before, except we now check if we encounter another instance of `Query`. If that's the case, the instance itself will take care of transforming its portion of the query, which might cause another instsance to be found...

Also note that if there is another instance of `Query`, we just ignore the `field_name` variable. This means that you could pass the internal query as `{'foo': query_2}`.

Unfortunately this does not work as expected:

{% highlight python %}
inner = Query(must={'text': ['san francisco', 'bay area']})
query_outer = Query(should={'query': inner, 'title': 'hiring'})
query_outer.to_elasticsearch()
>>> {
        'query': {
            'bool': {
                'should': [
                    {'match': {'title': 'hiring'}},
                    { 'query': {
                        'bool': {
                            'must': [
                                {'match': {'text': 'san francisco'}},
                                {'match': {'text': 'bay area'}}
                            ]
                        }
                    }
                }]
            }
        }
    }

{% endhighlight %}

Can you spot the mistake? The key `query` appears twice, so the previous is not a valid es query.

A little refactoring will bring us where we want.

{% highlight python %}
def expand_query(self):
    query = defaultdict(list)
    names = ['must', 'should', 'should_not']
    values = [self.must, self.should, self.should_not]
    for name, values in zip(names, values):
        if not values:
            continue
        for field_name, field_values in values.items():
            # field_name = 'text', field_values = ['san francisco', 'bay area']
            # OR
            # field_name = 'query',  field_values = some isntance of Query
            query[name].extend([
                    v.expand_query() if isinstance(v, Query)
                    else {'match': {field_name: v}}
                    for v in field_values])

    return {'bool': dict(query)}


def to_elasticsearch(self):
    return   {'query': self.expand_query()}
{% endhighlight %}

Let's test it:

{% highlight python %}
Query.expand_query = expand_query
Query.to_elasticsearch = to_elasticsearch
query_outer.to_elasticsearch()
>>> {
        'query': {
            'bool': {
                'should': [
                    {'match': {'title': 'hiring'}},
                    {'bool': {
                        'must': [
                            {'match': {'text': 'san francisco'}},
                            {'match': {'text': 'bay area'}}
                        ]
                    }}
                ]
            }
        }
    }

{% endhighlight %}


It worked!

The next step is to use this to increasing the quering power [for hackernews jobs index that we build earlier]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}).
