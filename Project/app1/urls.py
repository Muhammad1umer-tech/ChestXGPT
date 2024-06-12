# myapp/urls.py
from django.urls import path
from .views import PredictView
from django.contrib import admin
from django.urls import path, include
from django.contrib import admin
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('predict/', PredictView, name='predict'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
