---
layout: default
title: Blog
permalink: /blog/
---

{% for post in site.posts %}
  <div class="post">
    <!-- Displaying the date -->
    <p class="post-date">{{ post.date | date: "%B %d, %Y" }}</p>

    <!-- Displaying the title with underline -->
    <h2><a href="{{ post.url | relative_url }}" class="post-title">{{ post.title }}</a></h2>
    
    <p>{{ post.excerpt }}</p>
  </div>
{% endfor %}
