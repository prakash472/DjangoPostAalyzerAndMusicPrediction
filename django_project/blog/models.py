from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User
from django.urls import reverse

# class Category(models.Model):
#     title=models.CharField(max_length=50)
class Post(models.Model):
    title=models.CharField(max_length=100)
    content=models.TextField()
    date_posted=models.DateTimeField(default=timezone.now)
    review_positive=models.FloatField(default=0.0)
    author=models.ForeignKey(User, on_delete=models.CASCADE)
    # categories=models.ForeignKey(Category,on_delete=models.PROTECTED)
    

    def __str__(self):
        return self.title
    
    def get_absolute_url(self):
        return reverse("post-detail",kwargs={"pk":self.pk})
