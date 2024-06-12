from django.urls import path
from . import views
urlpatterns = [
    path('llm_model/', views.llm_model, name='llm_model'),
]