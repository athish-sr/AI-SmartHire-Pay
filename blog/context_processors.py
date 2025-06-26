# blog/context_processors.py
from .models import User
from django.db import connection
from django.shortcuts import render,redirect,get_object_or_404
def notification_count(request):
    if 'user_id' in request.session:
        user_id = request.session['user_id']
        user=get_object_or_404(User,pk=user_id)
        # Call the PostgreSQL function
        with connection.cursor() as cursor:
            cursor.execute("SELECT get_unread_notifications(%s)", [user.employee_id])
            count = cursor.fetchone()[0]  # Fetch the result from the query
    else:
        count = 0
        
    return {'unread_notification_count': count}

def salary_notification_count(request):
    if 'user_id' in request.session:
        user_id = request.session['user_id']
        user=get_object_or_404(User,pk=user_id)
        print(user.employee_id)
        # Call the PostgreSQL function
        with connection.cursor() as cursor:
            cursor.execute("SELECT get_salary_unread_notifications(%s)", [user.employee_id])
            count = cursor.fetchone()[0] 
            print(count) # Fetch the result from the query
    else:
        count = 0
    print(100)
    return {'salary_unread_notification_count': count}

def leave_notification_count(request):
    if 'user_id' in request.session:
        user_id = request.session['user_id']
        user=get_object_or_404(User,pk=user_id)
        # Call the PostgreSQL function
        with connection.cursor() as cursor:
            cursor.execute("SELECT get_leave_unread_notifications(%s)", [user.employee_id])
            count = cursor.fetchone()[0]  # Fetch the result from the query
    else:
        count = 0
        
    return {'leave_unread_notification_count': count}

def tax_notification_count(request):
    if 'user_id' in request.session:
        user_id = request.session['user_id']
        user=get_object_or_404(User,pk=user_id)
        # Call the PostgreSQL function
        with connection.cursor() as cursor:
            cursor.execute("SELECT get_tax_unread_notifications(%s)", [user.employee_id])
            count = cursor.fetchone()[0]  # Fetch the result from the query
    else:
        count = 0
        
    return {'tax_unread_notification_count': count}

