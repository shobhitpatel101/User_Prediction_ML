from django.urls import path
from . import views

urlpatterns=[
	path('',views.home,name='home'),
	path('foo',views.foo,name='foo'),
	path('nn',views.nn,name='nn'),
	path('ann',views.ann,name='ann'),
]
