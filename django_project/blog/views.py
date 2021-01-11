from django.shortcuts import render,get_object_or_404
from django.views.generic import ListView, DetailView, CreateView,UpdateView,DeleteView
from django.contrib.auth.mixins import LoginRequiredMixin,UserPassesTestMixin
from .models import Post
from django.contrib.auth.models import User

import requests

# Create your views here.
def home(request):
    sort_type=request.GET.get("sort_type")
    context={"posts":Post.objects.all(),
             "sort_type":sort_type}
    return render(request,"blog/home.html",context)

class PostListView(ListView):
    model=Post
    template_name= "blog/home.html"
    context_object_name= "posts"
    paginate_by = 5

    def get_queryset(self,*args, **kwargs):
        sort_type=self.request.GET.get("sort_type")
        if sort_type=="post_positive":
            return Post.objects.all().order_by("review_positive")
        elif sort_type=="post_negative":
            return Post.objects.all().order_by("-review_positive")
        else:
            return Post.objects.all().order_by("-date_posted")


class UserPostListView(ListView):
    model=Post
    template_name= "blog/user_posts.html"
    context_object_name= "posts"
    paginate_by = 2

    def get_queryset(self):
        user=get_object_or_404(User, username=self.kwargs.get("username"))
        return Post.objects.filter(author=user).order_by("-date_posted")

class PostDetailView(DetailView):
    model=Post

class PostCreateView(LoginRequiredMixin, CreateView):
    model=Post
    fields=["title","content"]

    def form_valid(self,form):
        URL = "http://127.0.0.1:5000/predict_review"
        form.instance.author=self.request.user
        initial_form=form.save(commit=False)
        form_content=initial_form.content
        TEXT_DATA = form_content
        review_text={"review": TEXT_DATA}
        response = requests.post(URL, json=review_text)
        data = response.json()["predictions"][0]
        form.instance.review_positive=data
        return super().form_valid(form)

class PostUpdateView(LoginRequiredMixin,UserPassesTestMixin,UpdateView):
    model=Post
    fields=["title","content"]

    def form_valid(self,form):
        form.instance.author=self.request.user
        return super().form_valid(form)
    
    def test_func(self):
        post=self.get_object()
        if self.request.user==post.author:
            return True
        return False

class PostDeleteView(LoginRequiredMixin, UserPassesTestMixin,DeleteView):
    model=Post
    success_url = "blog.home" 

    def test_func(self):
        post=self.get_object()
        if self.request.user==post.author:
            return True
        return False


def about(request):
    return render(request,"blog/about.html",{"title":"About"})

def post_check(request):
     return render(request,"blog/post_check.html")