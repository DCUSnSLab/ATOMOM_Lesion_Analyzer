from django.contrib import admin
from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    # path('coocr_upload', views.coocr_upload, name='coocr_upload'),
    # path('api', views.api, name='api'),
    # path('test', views.test, name='test'),
    # path('log', views.log, name='log'),
    # path('lesion', views.lesion, name='api2'),
    # path('silk/', include('silk.urls')),
]
