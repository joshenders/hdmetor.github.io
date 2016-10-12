---
layout: post
title: How to combine boolean queries in Elasticsearch
description: "How to combine elasticsearch queries."
modified: 2016-10-10
tags: [python, elasticsearch, boolean queries]
image:
  feature: abstract-10.jpg
  credit: dargadgetz
  creditlink: http://www.dargadgetz.com/ios-7-abstract-wallpaper-pack-for-iphone-5-and-ipod-touch-retina/
---

In a [previous post]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}) we saw how to use Elasticsearch to search for our dream job among the ones posted on hacker news.

Since Elasticsearch queries are basically JSON it"s really easy to lose track when we start nesting.

In this post we are going to define a Python class that will create the required query (read: JSON) on demand.

# Anatomy of an Elasticsearch query

A query in Elasticsearch in its general form is:

{% highlight python %}
{
   "query" : {
        "bool" : {
            "must" : [ ... ]
            "should" : [ ... ]
            "must_not" : [ ... ]
        }
    }
}

{% endhighlight %}

Where `must`, `should` and `must_not` correspond to the `AND`, `OR` and `NOT` operators respectively.

To use an [earlier example]({% post_url 2016-10-07-how-i-found-my-job-in-sf-using-hackernews-and-elasticsearch %}), the following query will require both "san francisco" and "machine learning" to be present in the `text` field.

{% highlight python %}
{
    "query" : {
        "bool" : {
            "must" : [
                {"match": {"text" : "san francisco"}},
                {"match": {"text" : "machine learning"}}
            ]
        }
    }
}
{% endhighlight %}

So far so good. But what if we want to nest a boolean query inside another boolean query?

![Deeper]({{site.url}}/assets/2016/deeper.jpg)

For example, what if we want to find something that matches ("san franisco" OR "bay area") AND ("machine learning" OR "data analysis")?

The resulting query will be something like this:

{% highlight python %}
{
    "query" : {
        "bool" : {
            "must" : [
                "bool" : {
                    "should" : [
                        {"match": {"text" : "san francisco"}},
                        {"match": {"text" : "bay area"}}
                    ]
                },
                "bool" : {
                    "should" : [
                        {"match": {"text" : "machine learning"}},
                        {"match": {"text" : "data analysis"}}
                    ]
                }
            ]
        }
    }
}
{% endhighlight %}

You can see how it"s easy to get lost here.

# Python to the rescue!

We want to represent the queries in a more human friendly way.

Before start coding, let's design the class. This will save so much time afterwards. What are the requirements, i.e. what is the expected behavior of our class (and its methods)?

- it should have arguments that correspond to the three boolean operators

- each operator can accept one or more term

- we need to be able to specify which field we want to search in

- each term can be another query

Let"s write down some obvious use cases.

{% highlight python %}
# easy case: one field, one term per field
query = Query(must={"text" : "san francisco"}, should_not={"text" : "new york"})

# one field, more terms
query_2 = Query(should={"text" : ["san francisco", "bay area"]})

# multiple fields, multiple terms
query_3 = Query(must={"text" : ["san francisco", "bay area"], "title" : "hiring"})

# query in a query
inner = query_2
query_outer = Query(must={"query" : inner, "title" : "hiring"})

{% endhighlight %}

Most important of all, our class needs to have a `to_elasticsearch` method to produce the desired json on demand.

Let"s start coding something that would work for the first query, and then let"s improve on that:

{% highlight python %}
class Query:

    def __init__(self, must=None, should=None, should_not=None):
        self.must = must
        self.should = should
        self.should_not = should_not
{% endhighlight %}

Since we are in the easy case, we can assume that the value passed are `dict`s.
In order to build the ES query, we then need to figure out which arguments have been passed (i.e. are not `None`) and put them in the query.

{% highlight python %}
    def to_elasticsearch(self):
        query = {name : value for name, value in zip(["must", "should", "should_not"], [self.must, self.should, self.should_not]) if value}
        query =  {
            "query" : {
                "bool" : query
            }
        }

        return query
{% endhighlight %}

So far so good. But what if the the same field has multiple values, as in `query_2`?
Our function needs to adapt. For one, we need to use the same key for all the values. So in the example above the key `text` has to be applied to both `"san francisco"` and `"bay area"`.

{% highlight python %}
    def to_elasticsearch(self):
    query = {}
        for name, values in zip(["must", "should", "should_not"], [self.must, self.should, self.should_not]):
            if not values:
                continue
            for field_name, field_values in values.items():
            # field_name = "text", field_values = ["san francisco", "bay area"]
                query[name] = [{"match" : {field_name : v}} for v in field_values]

            query =  {
                "query" : {
                    "bool" : query
                }
            }
        return query
{% endhighlight %}


Now we got it working for lists. What happens if we try mixed case like

{% highlight python %}
query_3 = Query(must={"text" : ["san francisco", "bay area"], "title" : hiring})
{% endhighlight %}

There are 2 things that will break:

- `query[name] = ...` will overwrite previous results (in our case the `title` field will overwirte the `text` one)

- `for v in field_values` part would not behave as expected (e.g. it will unpack a string)

To fix the first problem, we can just make `query` a `defaultdict` and `extend` it. To avoid problems communicating with the Elasticsearch client, we will convert back to a regular `dict` before returning.

{% highlight python %}
    from collections import defaultdict

    def to_elasticsearch(self):
    query = defaultdict(list)
        for name, values in zip(["must", "should", "should_not"], [self.must, self.should, self.should_not]):
            if not values:
                continue
            for field_name, field_values in values.items():
            # field_name = "text", field_values = ["san francisco", "bay area"]
                query[name].extend([{"match" : {field_name : v}} for v in field_values])

            query =  {
                "query" : {
                    "bool" : query
                }
            }
        return query
{% endhighlight %}


The most elegant way of solving the second problem, is to transform the input into a standard way. In that way the function that creates the query will be independent of the original input form.

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
{% endhighlight %}

    def preprocess(self, field):
        return {k: v if isinstance(v, list) else [v] for k,v in field.items()} of field else None

Some stuff to note here:

- if the `field` is `None`, we want to keep that way, not wrap it into a list. Returning `{}` would be fine too

- `[value]` is not the same as `list(value)`

- `query_1` output is now changed (there is an extra list), but it"s still a valid ES query, and it returns the same result.


# It"s turtles all the way down

Now the interesting part: we want to combine a [query withing a query](http://theinceptionbutton.com/).

Let"s do this step by step. First of all, we want the method `to_elasticsearch` to expand any inner queries. This is done by calling the same method on the inner queries. We will need to distinguish between a real `Query` object and just a query term.

The minimal edit seems to be something like this:

{% highlight python %}
    from collections import default dict

    def to_elasticsearch(self):
    query = defaultdict(list)
        for name, values in zip(["must", "should", "should_not"], [self.must, self.should, self.should_not]):
            if not values:
                continue
            for field_name, field_values in values.items():
            # field_name = "text", field_values = ["san francisco", "bay area"]
                query[name].extend([v.to_elasticsearch() if isinstance(v, Query) else {"match" : {field_name : v}} for v in field_values])

            query =  {
                "query" : {
                    "bool" : query
                }
            }
        return query
{% endhighlight %}

Just to reiterate: everything is the same, except we now check if we encounter another instance of `Query`. If that"s the case, the instance itself will take care of transforming its portion into an Elasticsearch query, unless there is another `Query` instance inside...

Unfortunately this does not work as expected:

{% highlight python %}
inner = Query(must={"text" : ["san francisco", "bay area"]})
query_outer = Query(should={"query" : inner, "title" : "hiring"})
query_outer.to_elasticsearch()
>>> {
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": "hiring"}},
                    { "query": {
                        "bool": {
                            "must": [
                                {"match": {"text": "san francisco"}},
                                {"match": {"text": "bay area"}}
                            ]
                        }
                    }
                }]
            }
        }
    }

{% endhighlight %}

Can you spot the mistake? The key `query` appears twice, so the previous is not a valid query. Just a little refactoring will bring us where we want.

{% highlight python %}
    def expand_query(self):

        query = defaultdict(list)

        for name, values in zip(["must", "should", "should_not"], [self.must, self.should, self.should_not]):
            if not values:
                continue

            for field_name, field_values in values.items():

            # field_name = "text", field_values = ["san francisco", "bay area"]
                query[name].extend([v.expand_query() if isinstance(v, Query) else {"match" : {field_name : v}} for v in field_values])

        return {"bool" : dict(query) }


    def to_elasticsearch(self):
        return   { "query" : self.expand_query() }
{% endhighlight %}

Let"s test it:

{% highlight python %}
inner = Query(must={"text" : ["san francisco", "bay area"]})
query_outer = Query(should={"query" : inner, "title" : "hiring"})
query_outer.to_elasticsearch()
>>> {
        "query": {
            "bool": {
                "should": [
                    {"match": {"title": "hiring"}},
                    {"bool": {
                        "must": [
                            {"match": {"text": "san francisco"}},
                            {"match": {"text": "bay area"}}
                        ]
                    }}
                ]
            }
        }
    }

{% endhighlight %}


It worked!
