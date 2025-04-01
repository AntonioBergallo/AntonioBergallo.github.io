---
layout: default
title: Blog
permalink: /blog/
---

{% for post in site.posts %}
  <div class="post">
    <!-- Displaying the title with underline -->
    <h2><a href="{{ post.url | relative_url }}" class="post-title">{{ post.title }}</a></h2>
    <!-- Displaying the date -->
    <p style="font-size: 0.8rem; color: #777; margin-bottom: 10px;">
      {{ post.date | date: "%B %d, %Y" }}
    </p>
    <p>{{ post.excerpt }}</p>
  </div>
{% endfor %}
