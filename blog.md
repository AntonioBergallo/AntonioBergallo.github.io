---
layout: default
title: Blog
permalink: /blog/
---

# Blog

Welcome to my blog! Here are all my posts:

{% for post in site.posts %}
  <div class="post">
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p>{{ post.excerpt }}</p>
  </div>
{% endfor %}
