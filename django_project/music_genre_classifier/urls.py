from django.urls import path
from . import views
urlpatterns = [
    path('predict_cnn/', views.predict, name='predict-cnn'),    
    path('',views.home, name='main-home'),
]