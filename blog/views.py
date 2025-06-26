import os
import secrets
import logging
import calendar
import joblib
import numpy as np
import pandas as pd
import matplotlib
from datetime import datetime
from io import BytesIO
from decimal import Decimal
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from fpdf import FPDF
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.utils.decorators import method_decorator
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse, JsonResponse
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.clickjacking import xframe_options_exempt
from django.views.decorators.http import require_POST
from django.utils import timezone
from django.utils.timezone import now
from django.utils.dateparse import parse_datetime
from django.urls import reverse
from django.template.loader import render_to_string
from django.core.mail import send_mail
from django.core.exceptions import ValidationError
from django.contrib.auth.password_validation import validate_password
from django.contrib import messages
from django.db import connection, IntegrityError
from django.contrib.auth.decorators import login_required
from django.contrib.auth.hashers import make_password,check_password
from django.forms.models import model_to_dict
from django.db.models import Sum
from .forms import (
    LoginForm, ForgotPasswordForm, EmployeeForm, LeaveRequestForm,
    TaxInputForm, SalaryPredictionForm
)

from .models import (
    User, Employee, Department, Designation, PasswordResetToken,
    LeaveRecord, Attendance, TaxInput, Salary, Payslip, notifications,Fund
)

logger = logging.getLogger(__name__)
loaded_model = joblib.load('model\predict_salary.pkl')
loaded_le_designation = joblib.load('model\designation_encoder.pkl')
loaded_le_education = joblib.load('model\education_encoder.pkl')

def login_view(request):
    login_form = LoginForm(request.POST or None)
    forgot_form = ForgotPasswordForm(request.POST or None)
    if request.method == "POST":
        if 'username' in request.POST:
            if login_form.is_valid():
                username = login_form.cleaned_data['username']
                password = login_form.cleaned_data['password']
                try:
                    if username == '1234' and password == '1234':
                        return redirect('upload_resume')
                    user = User.objects.get(username=username)
                    print(username,password)
                    print(make_password(password))
                    if check_password(password, user.password_hash):
                        if user.last_login is None:
                            # First-time login → Generate reset token & redirect
                            token = generate_password_reset_token(user)
                            reset_url = request.build_absolute_uri(
                                reverse('reset_password', kwargs={'token': token})
                            )
                            return redirect(reset_url)
                        user.last_login = now()
                        user.save()
                        request.session['user_id'] = user.user_id
                        messages.success(request, "Login successful.")
                        if user.role=="admin":
                            return redirect("/admin_dashboard")
                        elif user.role=='employee':
                            return redirect('emp_dashboard')
                    else:
                        messages.error(request, "Invalid username or password.")
                except User.DoesNotExist:
                    messages.error(request, "User does not exist.")
            else:
                messages.error(request, "Please correct the errors in the form.")
        elif 'employee_id' in request.POST:
            if forgot_form.is_valid():
                emp_id = forgot_form.cleaned_data['employee_id']
                email = forgot_form.cleaned_data['email']
                try:
                    user = User.objects.get(employee__employee_id=emp_id)
                    if user.employee.email == email:
                        token = generate_password_reset_token(user)
                        reset_url = request.build_absolute_uri(
                            reverse('reset_password', kwargs={'token': token})
                        )
                        send_mail(
                            subject="Reset your password",
                            message=f"Click the link to reset your password: {reset_url}",
                            from_email="technicalhealthguide360@gmail.com",
                            recipient_list=[user.employee.email],
                        )
                        messages.success(request, "Password reset link sent to your email.")
                    else:
                        messages.error(request, "Email does not match our records.")
                except (User.DoesNotExist, Employee.DoesNotExist):
                    messages.error(request, "Employee ID not found.")
            else:
                messages.error(request, "Invalid employee details.")
        return redirect("/")
    return render(request, "index.html", {
        'form': login_form,
        'forgot_form': forgot_form,
    })

def generate_password_reset_token(user):
    token = secrets.token_urlsafe(32)
    expiry = timezone.now() + timedelta(hours=1)
    PasswordResetToken.objects.create(user=user, token=token, expires_at=expiry)
    return token

def reset_password_view(request, token):
    try:
        reset_token = PasswordResetToken.objects.get(token=token)
    except PasswordResetToken.DoesNotExist:
        messages.error(request, "Invalid reset link.")
        return redirect("/")

    if not reset_token.is_valid():
        messages.error(request, "This password reset link has expired.")
        return redirect("/")

    if request.method == "POST":
        new_password = request.POST.get("new_password")
        confirm_password = request.POST.get('confirm_password')
        if new_password==confirm_password:
            try:
                validate_password(new_password)
            except ValidationError as e:
                err=""
                for error in e.messages:
                    err=err+"  "+error
                messages.error(request,err)
                return render(request, "reset_password_form.html")

            # Password is valid and confirmed
            reset_token.user.password_hash = make_password(new_password)
            messages.success(request, "Your password has been reset successfully.")

            if reset_token.user.last_login is None:
                request.session['user_id'] = reset_token.user.user_id
                reset_token.user.last_login = timezone.now()
                reset_token.user.save()
                reset_token.delete()
                if reset_token.user.role == "admin":
                    return redirect("admin_dashboard")
                elif reset_token.user.role == "employee":
                    return redirect("emp_dashboard")
            else:
                reset_token.user.save()
                reset_token.delete()
                return redirect("/")
        else:
            messages.error(request,"Password Does not Match.")

    return render(request, "reset_password_form.html")

def admin_home(request):
    return render(request,'admin_home.html')

def add_employee(request):
    if request.method == 'POST':
        form = EmployeeForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, 'Employee added successfully.')
            return redirect('view_employees')  # Redirect to a page showing all employees
        else:
            # Print and log specific error messages for each field
            for field, errors in form.errors.items():
                for error in errors:
                    print(f"Error in {field}: {error}")  # You can log this too if needed
            messages.error(request, 'Please correct the errors below.')
    
    else:
        form = EmployeeForm()
    return render(request, 'add_emp.html', {'form': form})

def get_designations(request):
    department_id = request.GET.get('department_id')
    designations = Designation.objects.filter(department_id=department_id).values('designation_id', 'designation_title')
    return JsonResponse(list(designations), safe=False)

def view_employees(request):
    employees = Employee.objects.select_related('department', 'designation').all().order_by('employee_id')
    return render(request, 'view_emp.html', {'employees': employees})

def search_employees(request):
    query = request.GET.get('query', '')
    results = []

    if query:
        employees = Employee.objects.filter(
            first_name__icontains=query
        ) | Employee.objects.filter(
            last_name__icontains=query
        ) | Employee.objects.filter(
            email__icontains=query
        ) | Employee.objects.filter(
            contact_info__icontains=query
        ).order_by('employee_id')
    else:
        employees = Employee.objects.all().order_by('employee_id')

    for emp in employees:
        results.append({
            'employee_id': emp.employee_id,
            'first_name': emp.first_name,
            'last_name': emp.last_name,
            'email': emp.email,
            'contact_info': emp.contact_info,
            'department': emp.department.department_name if emp.department else '',
            'designation': emp.designation.designation_title if emp.designation else ''
        })
    return JsonResponse({'results': results})

def edit_employee(request, employee_id):
    employee = get_object_or_404(Employee, employee_id=employee_id)

    if request.method == 'POST':
        form = EmployeeForm(request.POST, instance=employee)
        if form.is_valid():
            form.save()
            return redirect('view_employees')  # or some success URL
    else:
        form = EmployeeForm(instance=employee)

    return render(request, 'edit_employee.html', {'form': form, 'employee': employee})

def leave_requests(request):
    # Fetch all leave records
    leave_records = LeaveRecord.objects.filter(status="Pending")
    return render(request, 'leave_requests.html', {'leave_records': leave_records})

def approve_leave(request, leave_record_id):
    if request.method == 'POST':
        leave_record = LeaveRecord.objects.get(id=leave_record_id)
        status = request.POST.get('status')

        if status == 'Accepted':
            leave_record.status = 'Accepted'
        elif status == 'Rejected':
            leave_record.status = 'Rejected'

        # Assign approver if not already done
        if not leave_record.approved_by:
            approver = User.objects.get(user_id=request.session.get('user_id')) 
            leave_record.approved_by = approver

        leave_record.approved_on = timezone.now()
        leave_record.save()

        return redirect('leave_requests')  # Redirect back to leave requests list
    else:
        return redirect('leave_requests')  # For non-POST methods, redirect to leave requests

def upload_attendance(request):
    if request.method == 'POST':
        attendance_file = request.FILES.get('attendance_file')
        attendance_date = request.POST.get('attendance_date')

        if not attendance_file or not attendance_date:
            messages.error(request, "Please provide both a file and a date.")
            return redirect('upload_attendance')

        import pandas as pd
        df = pd.read_csv(attendance_file)
        print(df)

        with connection.cursor() as cursor:
            cursor.execute("SELECT check_attendance_eligibility(%s)", [attendance_date])
            result = cursor.fetchone()[0]
        print(result)
        if result != 'OK':
            messages.error(request, result)
            return redirect('upload_attendance')

        # ✅ Upload Attendance Records
        result = upload_attendance_csv(attendance_file)
        print(result)

        # ✅ Log Upload Entry
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO attendance_upload_log (attendance_date, uploaded_by, uploaded_at)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (attendance_date) DO NOTHING;
                """, [attendance_date,request.session.get('user_id'), timezone.now()])
        except Exception as e:
            messages.warning(request, f"Attendance uploaded, but failed to log upload: {str(e)}")
        else:
            messages.success(request, f"Attendance uploaded successfully for {attendance_date}.")

        return redirect('upload_attendance')

    return render(request, 'upload_attendance.html')

def upload_attendance_csv(file):
    import pandas as pd
    from .models import Attendance
    from django.db import IntegrityError

    inserted = 0
    errors = []

    try:
        file.seek(0)
        df = pd.read_csv(file)
    except Exception as e:
        return {'inserted': 0, 'skipped': 0, 'errors': [f'CSV Read Error: {str(e)}']}

    if df.empty:
        return {'inserted': 0, 'skipped': 0, 'errors': ['CSV is empty']}

    # Remove Leave records
    df = df[df['status'] != 'Leave']

    # Parse columns safely
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
    df['check_in_time'] = pd.to_datetime(df['check_in_time'], errors='coerce').dt.time
    df['check_out_time'] = pd.to_datetime(df['check_out_time'], errors='coerce').dt.time

    # Convert NaT to None
    df['check_in_time'] = df['check_in_time'].where(pd.notnull(df['check_in_time']), None)
    df['check_out_time'] = df['check_out_time'].where(pd.notnull(df['check_out_time']), None)

    for _, row in df.iterrows():
        try:
            Attendance.objects.update_or_create(
                employee_id=row['employee_id'],
                date=row['date'],
                defaults={
                    'status': row['status'],
                    'check_in_time': row['check_in_time'],
                    'check_out_time': row['check_out_time']
                }
            )
            inserted += 1
        except IntegrityError as e:
            errors.append(f"IntegrityError for employee {row['employee_id']}: {str(e)}")
        except Exception as e:
            errors.append(f"Error for employee {row['employee_id']}: {str(e)}")

    return {
        'inserted': inserted,
        'skipped': len(df) - inserted,
        'errors': errors
    }

def salary_prediction_view(request):
    salary = None
    error = None
    department = Department.objects.all()
    educations = loaded_le_education.classes_
    designations = loaded_le_designation.classes_

    if request.method == 'POST':
        form = SalaryPredictionForm(request.POST, educations=educations, designations=designations)
        if form.is_valid():
            try:
                experience = form.cleaned_data['experience']
                designation = form.cleaned_data['designation']
                education = form.cleaned_data['education']

                new_input = pd.DataFrame({
                    'experience': [experience],
                    'designation_encoded': loaded_le_designation.transform([designation]),
                    'education_encoded': loaded_le_education.transform([education])
                })

                pred_log = loaded_model.predict(new_input)
                pred_salary = np.expm1(pred_log)
                
                # Convert to a native Python float (not numpy float32)
                request.session['predicted_salary'] = float(pred_salary[0])  # Convert to regular float

                return redirect('predict_salary')  # Redirect to GET view
            except Exception as e:
                error = f"Error: {e}"
        else:
            error = "Invalid form input."
    else:
        form = SalaryPredictionForm(educations=educations, designations=designations)
        # Retrieve salary from session if available
        if 'predicted_salary' in request.session:
            salary = request.session.pop('predicted_salary')  # Remove after fetching

    return render(request, 'predict_salary.html', {
        'form': form,
        'salary': salary,
        'error': error,
        'department': department,
    })

def submit_leave_request(request):
    if request.method == 'GET':
        storage = messages.get_messages(request)
        list(storage)
    form = LeaveRequestForm()

    if request.method == 'POST':
        form = LeaveRequestForm(request.POST)
        if form.is_valid():
            leave_record = form.save(commit=False)

            user_id = request.session.get('user_id')
            print(user_id)
            try:
                # Get user and corresponding employee
                user = User.objects.get(user_id=user_id)
                employee = Employee.objects.get(employee_id=user.employee_id)
                leave_record.employee = employee
                leave_record.save()
                messages.success(request, "✅ Leave request submitted successfully!")
                return redirect('submit_leave_request')  # Redirect to clear form after success
            except (User.DoesNotExist, Employee.DoesNotExist):
                messages.error(request, "❌ Unable to locate employee record.")
        else:
            messages.error(request, "❌ Please correct the errors in the form.")

    return render(request, 'leave.html', {'form': form})

def tax_input_view(request):
    if request.method == 'GET':
        list(messages.get_messages(request))  # Clear flash messages

    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = get_object_or_404(User, pk=user_id)
    employee = get_object_or_404(Employee, pk=user.employee_id)

    today = timezone.now().date()

    try:
        tax_input = TaxInput.objects.filter(employee=employee).latest('created_at')

        # Determine financial year from tax_input.created_at
        created_at = tax_input.created_at.date()
        fy_start = created_at.year if created_at.month >= 4 else created_at.year - 1
        financial_year = f"{fy_start}-{fy_start + 1}"
        print(created_at, financial_year)

        # Check if we're in a new financial year window (April 1–15)
        if today.month == 4 and 1 <= today.day <= 15 and today.year > fy_start:
            # Pre-fill form with old tax data
            initial_data = model_to_dict(tax_input)
            initial_data.pop('id', None)  # Remove ID if present
            form = TaxInputForm(request.POST or None, initial=initial_data)

            if request.method == 'POST':
                if form.is_valid():
                    new_tax_data = form.save(commit=False)
                    new_tax_data.employee = employee
                    new_tax_data.created_at = timezone.now()
                    new_tax_data.save()
                    messages.success(request, "✅ Tax details submitted for new financial year.")
                    return redirect('tax_input')
            return render(request, 'tax_input.html', {
                'form': form,
                'previous_tax_input': tax_input,
                'resubmission': True
            })

        # Outside re-entry window: show read-only details
        return render(request, 'tax_details.html', {
            'tax_input': tax_input,
            'financial_year': financial_year
        })

    except TaxInput.DoesNotExist:
        # First-time submission
        form = TaxInputForm(request.POST or None)
        if request.method == 'POST':
            if form.is_valid():
                tax_data = form.save(commit=False)
                tax_data.employee = employee
                tax_data.created_at = timezone.now()
                tax_data.save()
                messages.success(request, "✅ Tax details submitted successfully.")
                return redirect('tax_input')
        return render(request, 'tax_input.html', {'form': form})


        # Else: show read-only existing tax details
        return render(request, 'tax_details.html', {'tax_input': tax_input})

    except TaxInput.DoesNotExist:
        # First-time submission
        form = TaxInputForm(request.POST or None)
        if request.method == 'POST':
            if form.is_valid():
                tax_data = form.save(commit=False)
                tax_data.employee = employee
                tax_data.created_at = timezone.now()
                tax_data.save()
                messages.success(request, "✅ Tax details submitted successfully.")
                return redirect('tax_input')
        return render(request, 'tax_input.html', {'form': form})

def emp_home(request):
    return render(request,'emp_home.html')

def summa(requests):
    inserted = 0
    errors = []

    try:
        df = pd.read_csv("C:\\Users\\athis\\Downloads\\final_tax_inputs_controlled.csv")
    except Exception as e:
        print(f"CSV Read Error: {str(e)}")

    if df.empty:
        print("CSV is empty")

    # Optional: convert date if needed
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    for _, row in df.iterrows():
        try:
            TaxInput.objects.create(
                employee_id=row['employee_id'],
                rent_paid=row['rent_paid']*12,
                lic=row['lic'],
                epf=row['epf'],
                home_principal=row['home_principal'],
                mutual_fund=row['mutual_fund'],
                nps_contribution=row['nps_contribution'],
                health_insurance=row['health_insurance'],
                parent_health=row['parent_health'],
                home_loan_interest=row['home_loan_interest'],
                education_expense=row['education_expense'],
                donations=row['donations'],
                created_at=row['created_at']  # optional; auto_now_add can take over
            )
            inserted += 1
        except Exception as e:
            print(f"Row {row['employee_id']} error: {str(e)}")

    return HttpResponse(inserted)
from django.shortcuts import render, redirect
from django.contrib import messages
from django.db import transaction
from .models import Fund
from datetime import date

def salary_operations_view(request):
    # Get the latest fund record
    latest_fund = Fund.objects.last()
    available_funds = latest_fund.available_funds
    # Calculate required funds (replace with your actual calculation)
    required_funds = Salary.objects.filter(payment_status='unpaid').aggregate(total_unpaid=Sum('net_salary'))['total_unpaid'] or 0.00  # Example value
    # Calculate difference
    difference = (latest_fund.available_funds if latest_fund else 0) - Decimal(required_funds)
    print(available_funds)
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'add_funds':
            try:
                amount = float(request.POST.get('amount', 0))
                print(amount)
                if amount <= 0:
                    messages.error(request, "Amount must be greater than zero")
                else:
                    with transaction.atomic():
                        # Get current available funds
                        current_funds = latest_fund.available_funds if latest_fund else 0
                        
                        # Create new fund record
                        Fund.objects.create(
                            credit=amount,
                            debit=0.00,
                        )
                        messages.success(request, f"Successfully added ₹{amount:.2f} to funds")
            except (ValueError, TypeError):
                messages.error(request, "Invalid amount entered")
            
            return redirect('salary_operations')
        
        elif action == 'generate':
            try:
                with connection.cursor() as cursor:
                    cursor.execute("CALL check_attendance_and_generate_salary()")
                messages.success(request, "✅ Salary generated for all employees.")
            except Exception as e:
                clean_message = str(e).split("CONTEXT:")[0].strip()  # Removes PostgreSQL context info
                messages.error(request, f"{clean_message}")

        elif action == 'transfer':
            if available_funds>=required_funds:
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("CALL transfer_salary()")
                    Fund.objects.create(
                            credit=0.00,
                            debit=required_funds) 
                    messages.success(request, "💸 Salaries transferred and logged in bank transactions.")
                except Exception as e:
                    clean_message = str(e).split("CONTEXT:")[0].strip()  # Removes PostgreSQL context info
                    messages.error(request, f"{clean_message}")
            else:
                messages.error(request,"⚠ Insufficient funds to transfer salary.")
        return redirect('salary_operations')
    
    context = {
        'latest_fund': latest_fund,
        'required_funds': required_funds,
        'difference': difference
    }
    return render(request, 'salary_operations.html', context)

def payslip_view(request,month,year):
    user_id = request.session.get('user_id')
    user = get_object_or_404(User, pk=user_id)
    employee = get_object_or_404(Employee, pk=user.employee_id)
    salary = Salary.objects.filter(employee=employee,salary_month=month,salary_year=year).latest('salary_month')

    pf = round(salary.basic_salary * Decimal('0.12'), 2)
    professional_tax = 200
    income_tax = round(salary.tax - professional_tax - pf, 2)
    total_earnings = salary.basic_salary + salary.allowances + salary.bonus
    total_deductions = salary.tax + salary.deductions
    deductions = salary.deductions
    month_name = calendar.month_name[salary.salary_month]
    context = {
        'employee': employee,
        'month':month_name,
        'salary': salary,
        'deductions': deductions,
        'pf': pf,
        'professional_tax': professional_tax,
        'income_tax': income_tax,
        'total_earnings': total_earnings,
        'total_deductions': total_deductions,
    }
    return render(request, 'payslip_template.html', context)

@csrf_exempt
@require_POST
def generate_payslip(request, emp_id, month, year):
    try:
        # 1. Get the employee
        employee = get_object_or_404(Employee, employee_id=emp_id)
        
        # 2. Get the uploaded PDF file (binary)
        pdf_file = request.FILES.get('pdf')

        if not pdf_file:
            return JsonResponse({'message': 'No PDF file received'}, status=400)
        
        # 3. Get the username of the person who generated it
        generated_by = request.session.get('user_id') or 'Unknown'

        # 4. Create a Payslip entry
        Payslip.objects.create(
            employee=employee,
            month=month,
            year=year,
            generated_by=generated_by,
            pdf_file=pdf_file.read()  # Read binary data from uploaded file
        )
        messages.success(request, 'Payslip has been generated and saved!')
        return JsonResponse({'success': True})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return JsonResponse({'succes': False})

def emp_dashboard(request):
    user = get_object_or_404(User, user_id=request.session.get('user_id'))
    employee = get_object_or_404(Employee, employee_id=user.employee_id)
    return render(request, 'emp_dashboard.html', {'employee': employee})

def profile_view(request):
    # Fetch the logged-in user's employee object
    employee = Employee.objects.get(email=request.user.email)  # Assuming the user email is linked to Employee's email

    # Get department and designation details
    department = employee.department.department_name if employee.department else 'Not Assigned'
    designation = employee.designation.designation_title if employee.designation else 'Not Assigned'
    
    # Calculate profile completion percentage
    fields = [
        employee.first_name,
        employee.last_name,
        employee.date_of_birth,
        employee.gender,
        employee.email,
        employee.contact_info,
        employee.address,
        employee.joining_date,
        department,
        designation,
        employee.bank_account_no,
        employee.pan_no,
        employee.status
    ]
    filled = sum([1 for field in fields if field])  # Count non-empty fields
    total = len(fields)
    profile_completion = (filled / total) * 100
    
    # Check for birthday or work anniversary today
    from datetime import date
    today = date.today()
    birthday_message = ''
    anniversary_message = ''

    if employee.date_of_birth.month == today.month and employee.date_of_birth.day == today.day:
        birthday_message = f"🎂 Happy Birthday, {employee.first_name}!"
    
    if employee.joining_date.month == today.month and employee.joining_date.day == today.day:
        anniversary_message = f"🎉 Happy Work Anniversary, {employee.first_name}!"

    # Pass data to the template
    context = {
        'employee': employee,
        'profile_completion': profile_completion,
        'birthday_message': birthday_message,
        'anniversary_message': anniversary_message,
        'department': department,
        'designation': designation
    }
    return render(request, 'profile_page.html', context)

from django.db.models.functions import ExtractMonth, ExtractYear
from django.db.models import F
from datetime import datetime

def attendance_view(request):
    user_id = request.session.get('user_id')
    user = get_object_or_404(User, pk=user_id)
    employee = get_object_or_404(Employee, pk=user.employee_id)

    # Get selected month/year from request or use current
    selected_month = request.GET.get('month')
    selected_year = request.GET.get('year')

    if selected_month and selected_year:
        selected_month = int(selected_month)
        selected_year = int(selected_year)
    else:
        from datetime import datetime
        now = datetime.now()
        selected_month = now.month
        selected_year = now.year

    available_periods = list(Attendance.objects.filter(employee=employee)
    .annotate(month=ExtractMonth('date'), year=ExtractYear('date'))
    .values('month', 'year').distinct().order_by('-year', '-month'))

    # Add month names for display
    for p in available_periods:
        p['month_name'] = calendar.month_name[p['month']]

    # Deduplicate years
    unique_years = sorted(set(p['year'] for p in available_periods), reverse=True)

    # Fetch filtered attendance
    attendance_records = Attendance.objects.filter(
        employee=employee,
        date__year=selected_year,
        date__month=selected_month
    ).order_by('-date')

    # Leave records (no filtering needed)
    leave_records = LeaveRecord.objects.filter(
        employee=employee
    ).order_by('-start_date')

    with connection.cursor() as cursor:
        cursor.execute("SELECT get_sick_leave_days(%s, %s);", [employee.employee_id, selected_year])
        sick_leave_days = cursor.fetchone()[0] or 0

        cursor.execute("SELECT get_casual_leave_days(%s, %s);", [employee.employee_id, selected_year])
        casual_leave_days = cursor.fetchone()[0] or 0

    sick_leave_days_remaining = max(15 - sick_leave_days, 0)
    casual_leave_days_remaining = max(15 - casual_leave_days, 0)

    return render(request, 'attendance.html', {
    'attendance_records': attendance_records,
    'leave_records': leave_records,
    'sick_leave_days': sick_leave_days_remaining,
    'casual_leave_days': casual_leave_days_remaining,
    'available_periods': available_periods,
    'unique_years': unique_years,
    'selected_month': selected_month,
    'selected_year': selected_year
})

def salary_details(request):
    user_id = request.session.get('user_id')
    user = get_object_or_404(User, pk=user_id)
    employee = get_object_or_404(Employee, pk=user.employee_id)
    
    salary = Salary.objects.filter(employee=employee).latest('salary_month')
    
    pf = round(salary.basic_salary * Decimal('0.12'), 2)
    professional_tax = 200
    income_tax = round(salary.tax - professional_tax - pf, 2)
    total_earnings = salary.basic_salary + salary.allowances + salary.bonus
    total_deductions = salary.tax + salary.deductions
    deductions = salary.deductions
    month_name = calendar.month_name[salary.salary_month]
    from datetime import datetime
    current_year = datetime.now().year

    # Get annual salary data
    with connection.cursor() as cursor:
        cursor.execute("SELECT * FROM get_annual_salary_details(%s, %s);", [user.employee_id, current_year])
        row = cursor.fetchone()

    if row:
        annual_basic_salary = row[0]
        annual_earnings = row[1]
        annual_net_salary = row[2]
        annual_tax_paid = row[3]
        annual_bonus = row[4]
        annual_deductions = row[5]
    else:
        annual_basic_salary = annual_net_salary = annual_tax_paid = annual_bonus = annual_deductions = 0
        annual_earnings = 0
    
    annual_pf = annual_basic_salary * Decimal('0.12')

    # ✅ Get distinct available years for this employee
    available_years = Salary.objects.filter(employee=employee).values_list('salary_year', flat=True).distinct().order_by('-salary_year')
    selected_year = request.GET.get('year', current_year)
    print(selected_year)
    salaries = Salary.objects.filter(employee=employee, salary_year=selected_year).order_by('-salary_month')
    
    for s in salaries:
        s.month_name = calendar.month_name[s.salary_month]
    context = {
        'employee': employee,
        'month': month_name,
        'salary': salary,
        'deductions': deductions,
        'pf': pf,
        'professional_tax': professional_tax,
        'income_tax': income_tax,
        'total_earnings': total_earnings,
        'total_deductions': total_deductions,
        'annual_net_salary': annual_net_salary,
        'annual_pf': annual_pf,
        'annual_tax_paid': annual_tax_paid - annual_pf,
        'annual_bonus': annual_bonus,
        'annual_deduction': annual_deductions,
        'annual_earnings': annual_earnings,
        'available_years': available_years,
        'salaries':salaries,
        'selected_year':selected_year
    }

    return render(request, 'salary.html', context)

def notify(request):
    user_id=request.session.get('user_id')
    user=get_object_or_404(User,pk=user_id)
    employee=get_object_or_404(Employee,pk=user.employee_id)
    salary_notifications = notifications.objects.filter(employee=employee, category='salary').order_by('-timestamp')
    leave_notifications = notifications.objects.filter(employee=employee, category='leave').order_by('-timestamp')
    tax_notifications = notifications.objects.filter(employee=employee, category='tax').order_by('-timestamp')

    context = {
        'salary_notifications': salary_notifications,
        'leave_notifications': leave_notifications,
        'tax_notifications': tax_notifications,
    }
    return render(request, 'notify.html', context)

def mark_notification_as_read(request, notification_id):
    # Get the notification by its ID
    notification = get_object_or_404(notifications, id=notification_id)
    notification.is_read = True
    notification.save()
    return redirect('notify')  # Adjust the redirect as needed

def category_notifications(request, category):
    user_id=request.session.get('user_id')
    user=get_object_or_404(User,pk=user_id)
    employee=get_object_or_404(Employee,pk=user.employee_id)
    notification = notifications.objects.filter(employee=employee,category=category)
    return render(request, 'category_notifications.html', {'notifications': notification, 'category': category})

def mark_all_as_read(request, category):
    # Mark notifications as read based on category
    if category == 'salary':
        notification = notifications.objects.filter(category='salary', is_read=False)
    elif category == 'leave':
        notification = notifications.objects.filter(category='leave', is_read=False)
    elif category == 'tax':
        notification = notifications.objects.filter(category='tax', is_read=False)
    else:
        return redirect('notify')  # Invalid category
    
    # Mark all notifications as read
    notification.update(is_read=True)
    
    # Redirect back to the notification page
    return redirect('notify')

@method_decorator(xframe_options_exempt, name='dispatch')
class DownloadSalaryReportPDF(View):
    def get(self, request, *args, **kwargs):
        try:
            response = get_report_data(request)
            if not isinstance(response, HttpResponse):
                raise ValueError("Invalid response from report generation")
                
            response['Content-Type'] = 'application/pdf'
            response['Content-Disposition'] = 'attachment; filename="salary_report.pdf"'
            response['X-Content-Type-Options'] = 'nosniff'
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate PDF: {str(e)}")
            return HttpResponse(f"Failed to generate PDF: {str(e)}", status=500)

def generate_salary_report(employee_data, salary_data, leave_data, output_filename):
    """
    Generates a professional salary report PDF with the provided data
    Returns HttpResponse with PDF content
    """
    # ======================
    # 1. DATA PROCESSING
    # ======================
    df = pd.DataFrame(salary_data)
    current_month = datetime.now().strftime("%B")
    
    # Calculate derived fields
    df["Gross"] = df["Basic"] + df["HRA"] + df["Bonus"]
    df["Deductions"] = df["Professional_Tax"] + df["TDS"] + df["PF"] 
    df["Net"] = df["Gross"] - df["Deductions"]
    
    # Calculate YTD totals
    ytd_totals = {
        "Basic": df["Basic"].sum(),
        "HRA": df["HRA"].sum(),
        "Bonus": df["Bonus"].sum(),
        "Gross": df["Gross"].sum(),
        "Professional_Tax": df["Professional_Tax"].sum(),
        "TDS": df["TDS"].sum(),
        "PF": df["PF"].sum(),
        "Deductions": df["Deductions"].sum(),
        "Net": df["Net"].sum()
    }

    # ======================
    # 2. CHART GENERATION
    # ======================
    def create_charts():
        # Create temporary files for charts
        pie_path = f"temp_pie_{employee_data['emp_id']}.png"
        trend_path = f"temp_trend_{employee_data['emp_id']}.png"
        
        # Salary Composition Pie Chart
        plt.figure(figsize=(6, 6))
        components = ["Basic", "HRA", "Bonus"]
        values = [ytd_totals["Basic"], ytd_totals["HRA"], ytd_totals["Bonus"]]
        plt.pie(values, labels=components, autopct='%1.1f%%')
        plt.title("Salary Composition (YTD)")
        plt.savefig(pie_path)
        plt.close()
        
        # Monthly Trend Bar Chart
        plt.figure(figsize=(10, 6))
        months = df["Month"]
        x = np.arange(len(months))
        width = 0.35
        
        plt.bar(x - width/2, df["Gross"], width, label='Gross Salary')
        plt.bar(x + width/2, df["Net"], width, label='Net Salary')
        plt.xlabel('Month')
        plt.ylabel('Amount (Rs.)')
        plt.title('Monthly Salary Trend')
        plt.xticks(x, months)
        plt.legend()
        plt.tight_layout()
        plt.savefig(trend_path, bbox_inches='tight')
        plt.close()
        
        return pie_path, trend_path

    pie_path, trend_path = create_charts()

    # ======================
    # 3. PDF GENERATION
    # ======================
    class ReportPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                # Cover Page
                self.set_font('Arial', 'B', 24)
                self.cell(0, 40, "ABC Company", 0, 1, 'C')
                self.set_font('Arial', 'B', 18)
                self.cell(0, 10, "Employee Annual Salary Report - 2024", 0, 1, 'C')
                self.set_font('Arial', 'I', 12)
                self.cell(0, 10, "(Confidential Document)", 0, 1, 'C')
                self.ln(20)
                
                # Cover page content
                self.set_fill_color(200, 220, 255)
                self.rect(50, 100, 110, 80, 'F')
                self.set_font('Arial', '', 14)
                self.set_xy(50, 110)
                self.cell(110, 10, f"Prepared for: {employee_data['name']}", 0, 1, 'C')
                self.set_xy(50, 130)
                self.cell(110, 10, f"Report Period: January 2025 - {current_month} 2025", 0, 1, 'C')
                self.set_xy(50, 150)
                self.cell(110, 10, f"Generated On: {datetime.now().strftime('%d-%b-%Y')}", 0, 1, 'C')
                self.add_page()

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

        def add_chapter_title(self, title):
            self.set_font('Arial', 'B', 14)
            self.set_fill_color(200, 220, 255)
            self.cell(0, 10, title, 0, 1, 'L', 1)
            self.ln(4)

        def create_data_table(self, headers, data, col_widths):
            self.set_font('Arial', 'B', 10)
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 10, header, 1, 0, 'C', 1)
            self.ln()
            
            self.set_font('Arial', '', 10)
            for row in data:
                for i, item in enumerate(row):
                    self.cell(col_widths[i], 10, str(item), 1)
                self.ln()

    # Initialize PDF buffer
    buffer = BytesIO()
    pdf = ReportPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ======================
    # 4. REPORT CONTENT
    # ======================
    # Employee Details
    pdf.add_chapter_title("1. Employee Details")
    emp_details = [
        ["Employee Name", employee_data["name"]],
        ["Employee ID", employee_data["emp_id"]],
        ["Employee Email", employee_data["email"]],
        ["Designation", employee_data["designation"]],
        ["Department", employee_data["department"]],
        ["Contact Number", employee_data["phone"]],
        ["Location", employee_data["location"]],
        ["Joining Date", employee_data["joining_date"]],
        ["Pan Number", employee_data["pan"]],
        ["Bank Account", employee_data["bank_account"]]
    ]
    pdf.create_data_table(["Field", "Details"], emp_details, [60, 130])

    # Salary Breakdown
    pdf.add_page()
    pdf.add_chapter_title("2. Salary Breakdown")
    
    # Salary Components Table
    salary_table = []
    for _, row in df.iterrows():
        salary_table.append([
            row["Month"],
            f"Rs.{row['Basic']:,}",
            f"Rs.{row['HRA']:,}",
            f"Rs.{row['Bonus']:,}",
            f"Rs.{row['Gross']:,}"
        ])
    pdf.create_data_table(
        ["Month", "Basic", "HRA", "Bonus", "Gross"], 
        salary_table, 
        [30, 30, 30, 30, 40]
    )

    # Deductions Table
    pdf.ln(10)
    deductions_table = []
    for _, row in df.iterrows():
        deductions_table.append([
            row["Month"],
            f"Rs.{row['Professional_Tax']:,}",
            f"Rs.{row['TDS']:,}",
            f"Rs.{row['PF']:,}",
            f"Rs.{row['Deductions']:,}"
        ])
    pdf.create_data_table(
        ["Month", "Professional Tax", "TDS", "PF", "Total Deductions"], 
        deductions_table, 
        [25, 35, 25, 35, 35]
    )

    # Net Salary Table
    pdf.ln(10)
    net_salary_table = []
    for _, row in df.iterrows():
        net_salary_table.append([
            row["Month"],
            f"Rs.{row['Gross']:,}",
            f"Rs.{row['Deductions']:,}",
            f"Rs.{row['Net']:,}"
        ])
    pdf.create_data_table(
        ["Month", "Gross Salary", "Total Deductions", "Net Salary"], 
        net_salary_table, 
        [40, 40, 40, 40]
    )

    # Graphical Analysis
    pdf.add_page()
    pdf.add_chapter_title("3. Graphical Analysis")

    # Salary Composition
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Salary Composition", 0, 1)
    pdf.image(pie_path, x=50, w=110)

    # Monthly Trend
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, "Monthly Salary Trend", 0, 1)
    pdf.image(trend_path, x=10, w=190)

    # Leave Details
    pdf.add_page()
    pdf.add_chapter_title("4. Leave Details")
    leave_table = []
    for i in range(len(leave_data["Leave_Type"])):
        leave_table.append([
            leave_data["Leave_Type"][i],
            str(leave_data["Entitled"][i]),
            str(leave_data["Availed"][i]),
            str(leave_data["Balance"][i])
        ])
    pdf.create_data_table(
        ["Leave Type", "Entitled", "Availed", "Balance"], 
        leave_table, 
        [50, 30, 30, 30]
    )

    # PF Summary
    pdf.add_chapter_title("5. Provident Fund (PF) Summary")
    pf_headers = ["Month", "Employee Contribution (Rs)", "Employer Contribution (Rs)", "Total (Rs)"]
    pf_data = []
    for _, row in df.iterrows():
        pf_data.append([
            row["Month"],
            f"Rs.{row['PF']:,}",
            f"Rs.{row['PF']:,}",
            f"Rs.{row['PF']*2:,}"
        ])
    pdf.create_data_table(pf_headers, pf_data, [30, 50, 50, 40])

    # Tax Summary
    pdf.add_page()
    pdf.add_chapter_title("6. Tax Summary")
    tax_headers = ["Component", "YTD Amount (Rs)"]
    tax_data = [
        ["Total Taxable Income", f"Rs.{ytd_totals['Gross']:,}"],
        ["TDS Deducted", f"Rs.{ytd_totals['TDS']:,}"]
    ]
    pdf.create_data_table(tax_headers, tax_data, [70, 70])
    pdf.ln(5)
    # Notes
    pdf.add_chapter_title("7. Notes")
    notes = [
        "1. All figures are pre-audit and subject to revision during tax filing.",
        "2. Unpaid leave deductions calculated at Rs3,000/day.",
        "3. PF contributions include voluntary top-up of Rs2,500/month.",
        "4. This document is system generated and doesn't require physical signature."
    ]
    pdf.set_font('Arial', '', 8)
    for note in notes:
        pdf.cell(0, 10, note, 0, 1)

    # Generate PDF to buffer
    pdf.output(buffer)
    buffer.seek(0)

    # Create response
    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/pdf'
    )
    response['Content-Disposition'] = f'attachment; filename="{output_filename}"'
    
    # Clean up temporary files
    for temp_file in [pie_path, trend_path]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    return response

def get_report_data(request):
    """
    Get salary report data for an employee
    Returns HttpResponse with PDF content
    """
    try:
        user_id = request.session.get('user_id')
        user = get_object_or_404(User, pk=user_id)
        employee = get_object_or_404(Employee, pk=user.employee_id)
        current_year = timezone.now().year

        # Prepare employee data
        employee_details = {
            "name": f"{employee.first_name} {employee.last_name}",
            "emp_id": f"EMP-{employee.employee_id:04d}",
            "email": employee.email,
            "phone": employee.contact_info,
            "pan": employee.pan_no,
            "designation": employee.designation.designation_title if employee.designation else "Not Specified",
            "department": employee.department.department_name if employee.department else "Not Specified",
            "location": "Coimbatore",
            "joining_date": employee.joining_date.strftime("%d-%b-%Y"),
            "bank_account": employee.bank_account_no or "Not Provided"
        }

        # Get salary data
        salaries = Salary.objects.filter(
            employee=employee,
            salary_year=current_year
        ).order_by('salary_month')

        salary_data = {
            "Month": [],
            "Basic": [],
            "HRA": [],
            "Bonus": [],
            "Professional_Tax": [],
            "TDS": [],
            "PF": [],
            "Gross": [],
            "Net": []
        }

        for salary in salaries:
            month_name = calendar.month_name[int(salary.salary_month)]
            salary_data["Month"].append(month_name)
            salary_data["Basic"].append(float(salary.basic_salary))
            salary_data["HRA"].append(float(salary.allowances))
            salary_data["Bonus"].append(float(salary.bonus))
            salary_data["Professional_Tax"].append(200)
            salary_data["TDS"].append(float(salary.tax - (salary.basic_salary * Decimal("0.12")) - Decimal("200")))
            salary_data["PF"].append(float(salary.basic_salary * Decimal("0.12")))
            salary_data["Gross"].append(float(salary.basic_salary + salary.allowances + salary.bonus))
            salary_data["Net"].append(float(salary.net_salary))

        # Get leave data
        leaves = LeaveRecord.objects.filter(
            employee=employee,
            status='Accepted',
            start_date__year=current_year
        )

        LEAVE_ENTITLEMENTS = {
            'Paid Leave': 15,
            'Sick Leave': 15,
            'Unpaid Leave': 'N/A',
            'Privilege Leave': 12,
            'Maternity Leave': 180
        }

        leave_data = {
            "Leave_Type": [],
            "Entitled": [],
            "Availed": [],
            "Balance": []
        }

        availed_days = {leave_type: 0 for leave_type in LEAVE_ENTITLEMENTS}
        
        for leave in leaves:
            if leave.leave_type == 'Unauthorized Leave':
                duration = (leave.end_date - leave.start_date).days + 1
                availed_days['Unpaid Leave'] += duration
            elif leave.leave_type == 'Casual Leave':
                duration = (leave.end_date - leave.start_date).days + 1
                availed_days['Paid Leave'] += duration
            else:
                duration = (leave.end_date - leave.start_date).days + 1
                availed_days[leave.leave_type] += duration
        
        for leave_type, entitled in LEAVE_ENTITLEMENTS.items():
            availed = availed_days[leave_type]
            leave_data["Leave_Type"].append(leave_type)
            leave_data["Entitled"].append(entitled)
            leave_data["Availed"].append(availed)
            leave_data["Balance"].append('N/A' if entitled == 'N/A' else (entitled - availed if availed <= entitled else 0))

        return generate_salary_report(
            employee_data=employee_details,
            salary_data=salary_data,
            leave_data=leave_data,
            output_filename=f"Salary_Report_{employee_details['emp_id']}.pdf"
        )

    except Exception as e:
        logger.error(f"Report data error: {str(e)}")
        return HttpResponse(f"Error generating report: {str(e)}", status=400)
    
import pandas as pd
from datetime import datetime
from django.http import HttpResponse
from io import BytesIO
from django.views import View
from django.utils import timezone
from django.shortcuts import get_object_or_404
import calendar

class DownloadSalaryReportExcel(View):
    def get(self, request, *args, **kwargs):
        try:
            # Get employee data
            user_id = request.session.get('user_id')
            user = get_object_or_404(User, pk=user_id)
            employee = get_object_or_404(Employee, pk=user.employee_id)
            current_year = timezone.now().year
            current_month = timezone.now().month

            # Prepare employee details (DICTIONARY)
            employee_dict = {
                "Employee Name": f"{employee.first_name} {employee.last_name}",
                "Employee ID": f"EMP-{employee.employee_id:04d}",
                "Designation": employee.designation.designation_title if employee.designation else "N/A",
                "Department": employee.department.department_name if employee.department else "N/A",
                "Joining Date": employee.joining_date.strftime("%d-%b-%Y"),
                "Bank Account": employee.bank_account_no or "N/A",
                "PAN Number": employee.pan_no or "N/A",
                "Report Period": f"Jan {current_year} - {datetime.now().strftime('%b %Y')}",
                "Generated On": datetime.now().strftime("%d-%b-%Y %H:%M")
            }

            # Get salary data for current year up to current month
            salaries = Salary.objects.filter(
                employee=employee,
                salary_year=current_year,
                salary_month__lte=current_month
            ).order_by('salary_month')

            # Prepare salary data
            salary_data = []
            for salary in salaries:
                salary_data.append({
                    "Month": calendar.month_name[salary.salary_month],
                    "Basic Salary": float(salary.basic_salary),
                    "HRA": float(salary.allowances),
                    "Bonus": float(salary.bonus),
                    "Professional Tax": 200.00,
                    "TDS": float(salary.tax - (salary.basic_salary * Decimal("0.12")) - Decimal("200")),
                    "PF": float(salary.basic_salary * Decimal("0.12")),
                    "Gross Salary": float(salary.basic_salary + salary.allowances + salary.bonus),
                    "Total Deductions": float(200 + (salary.tax - (salary.basic_salary * Decimal("0.12")) - Decimal("200")) + (salary.basic_salary * Decimal("0.12"))),
                    "Net Salary": float(salary.net_salary)
                })

            # Calculate YTD totals
            df = pd.DataFrame(salary_data)
            ytd_totals = {
                "Basic Salary": df["Basic Salary"].sum(),
                "HRA": df["HRA"].sum(),
                "Bonus": df["Bonus"].sum(),
                "Professional Tax": df["Professional Tax"].sum(),
                "TDS": df["TDS"].sum(),
                "PF": df["PF"].sum(),
                "Gross Salary": df["Gross Salary"].sum(),
                "Total Deductions": df["Total Deductions"].sum(),
                "Net Salary": df["Net Salary"].sum()
            }

            # Get leave data
            leaves = LeaveRecord.objects.filter(
                employee=employee,
                status='Accepted',
                start_date__year=current_year
            )

            LEAVE_ENTITLEMENTS = {
                'Paid Leave': 15,
                'Sick Leave': 15,
                'Unpaid Leave': 'N/A',
                'Privilege Leave': 12,
                'Maternity Leave': 180
            }

            leave_data = []
            availed_days = {leave_type: 0 for leave_type in LEAVE_ENTITLEMENTS}
            
            for leave in leaves:
                if leave.leave_type == 'Unauthorized Leave':
                    duration = (leave.end_date - leave.start_date).days + 1
                    availed_days['Unpaid Leave'] += duration
                elif leave.leave_type == 'Casual Leave':
                    duration = (leave.end_date - leave.start_date).days + 1
                    availed_days['Paid Leave'] += duration
                else:
                    duration = (leave.end_date - leave.start_date).days + 1
                    availed_days[leave.leave_type] += duration
            
            for leave_type, entitled in LEAVE_ENTITLEMENTS.items():
                availed = availed_days[leave_type]
                balance = 'N/A' if entitled == 'N/A' else (entitled - availed if availed <= entitled else 0)
                leave_data.append({
                    "Leave Type": leave_type,
                    "Entitled": entitled,
                    "Availed": availed,
                    "Balance": balance
                })

            # Create Excel file in memory
            output = BytesIO()
            writer = pd.ExcelWriter(output, engine='xlsxwriter')
            workbook = writer.book

            # ========== Formatting Styles ==========
            # Company header style
            company_header_format = workbook.add_format({
                'bold': True,
                'font_size': 20,
                'align': 'center',
                'valign': 'vcenter',
                'font_color': '#1F497D',
                'font_name': 'Calibri',
                'bottom': 1
            })

            # Report title style
            report_title_format = workbook.add_format({
                'bold': True,
                'font_size': 16,
                'align': 'center',
                'valign': 'vcenter',
                'font_color': '#1F497D',
                'font_name': 'Calibri',
                'bottom': 2
            })

            # Employee info label style
            employee_info_label_format = workbook.add_format({
                'bold': True,
                'font_size': 11,
                'font_color': '#1F497D',
                'font_name': 'Calibri',
                'align': 'left',
                'border': 0
            })

            # Employee info value style
            employee_info_value_format = workbook.add_format({
                'font_size': 11,
                'font_name': 'Calibri',
                'align': 'left',
                'border': 0
            })

            # Section header style
            section_header_format = workbook.add_format({
                'bold': True,
                'font_size': 12,
                'align': 'left',
                'valign': 'vcenter',
                'font_color': '#1F497D',
                'font_name': 'Calibri',
                'top': 1,
                'bottom': 1
            })

            # Column header style
            header_format = workbook.add_format({
                'bold': True,
                'align': 'center',
                'valign': 'vcenter',
                'fg_color': '#1F497D',
                'font_color': 'white',
                'border': 1,
                'text_wrap': True,
                'font_name': 'Calibri'
            })

            # Data cell style
            data_cell_format = workbook.add_format({
                'font_name': 'Calibri',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            # Currency format
            currency_format = workbook.add_format({
                'num_format': '₹#,##0.00',
                'align': 'right',
                'font_name': 'Calibri',
                'border': 1
            })

            # Leave data format
            leave_data_format = workbook.add_format({
                'font_name': 'Calibri',
                'border': 1,
                'align': 'center',
                'valign': 'vcenter'
            })

            # Summary format
            summary_format = workbook.add_format({
                'bold': True,
                'fg_color': '#DCE6F1',
                'border': 1,
                'num_format': '₹#,##0.00',
                'font_name': 'Calibri',
                'align': 'right'
            })

            # Confidential notice format
            confidential_format = workbook.add_format({
                'bold': True,
                'font_color': '#C00000',
                'align': 'center',
                'font_name': 'Calibri',
                'border': 0
            })

            # ========== Single Sheet Report ==========
            # Start writing at row 0
            worksheet = workbook.add_worksheet('Salary Report')
            row = 0
            
            # 1. Company Header
            worksheet.merge_range(row, 0, row, 9, "ABC COMPANY", company_header_format)
            worksheet.set_row(row, 30)
            row += 1
            
            # 2. Report Title
            worksheet.merge_range(row, 0, row, 9, "ANNUAL SALARY STATEMENT", report_title_format)
            worksheet.set_row(row, 25)
            row += 2  # Extra space after title
            
            # 3. Employee Information Section
            worksheet.write(row, 0, "Employee Information", section_header_format)
            worksheet.set_row(row, 20)
            row += 1
            
            # Employee Details List (CORRECTED VARIABLE NAME)
            employee_info_entries = [
                ["Name:", employee_dict["Employee Name"]],
                ["Employee ID:", employee_dict["Employee ID"]],
                ["Designation:", employee_dict["Designation"]],
                ["Department:", employee_dict["Department"]],
                ["Joining Date:", employee_dict["Joining Date"]],
                ["PAN:", employee_dict["PAN Number"]],
                ["Bank A/C:", employee_dict["Bank Account"]],
                ["Report Period:", employee_dict["Report Period"]],
                ["Generated On:", employee_dict["Generated On"]]
            ]
            
            for detail in employee_info_entries:
                worksheet.write(row, 0, detail[0], employee_info_label_format)
                worksheet.write(row, 1, detail[1], employee_info_value_format)
                row += 1
            row += 1  # Extra space after employee info
            
            # 4. Confidential Notice
            worksheet.merge_range(row, 0, row, 9, "CONFIDENTIAL - FOR AUTHORIZED PERSONNEL ONLY", confidential_format)
            row += 2  # Extra space after notice
            
            # 5. Salary Breakdown Section
            worksheet.write(row, 0, "Monthly Salary Breakdown", section_header_format)
            row += 1
            
            # Salary Table Headers
            salary_headers = ["Month", "Basic Salary", "HRA", "Bonus", "Professional Tax", 
                            "TDS", "PF", "Gross Salary", "Total Deductions", "Net Salary"]
            
            for col, header in enumerate(salary_headers):
                worksheet.write(row, col, header, header_format)
                # Set column widths
                if col == 0:  # Month column
                    worksheet.set_column(col, col, 15)
                else:  # All other columns
                    worksheet.set_column(col, col, 12)
            
            row += 1
            
            # Salary Data Rows
            for month_data in salary_data:
                for col, (key, value) in enumerate(month_data.items()):
                    if col == 0:  # Month name
                        worksheet.write(row, col, value, data_cell_format)
                    else:  # Numeric values
                        worksheet.write(row, col, value, currency_format)
                row += 1
            
            # YTD Totals Row
            worksheet.write(row, 0, "Year-to-Date Totals", workbook.add_format({
                'bold': True,
                'font_name': 'Calibri',
                'border': 1,
                'align': 'right',
                'fg_color': '#DCE6F1'
            }))
            
            for col in range(1, len(salary_headers)):
                worksheet.write(row, col, ytd_totals.get(salary_headers[col], ""), summary_format)
            
            row += 2  # Extra space after salary table
            
            # 6. Leave Summary Section
            worksheet.write(row, 0, "Leave Summary", section_header_format)
            row += 1
            
            # Leave Table Headers
            leave_headers = ["Leave Type", "Entitled", "Availed", "Balance"]
            for col, header in enumerate(leave_headers):
                worksheet.write(row, col, header, header_format)
                worksheet.set_column(col, col, 15)  # Set leave columns to 15 width
            
            row += 1
            
            # Leave Data Rows
            for leave_record in leave_data:
                for col, (key, value) in enumerate(leave_record.items()):
                    cell_format = leave_data_format
                    if col == 3 and value == 0:  # Highlight zero balance
                        cell_format = workbook.add_format({
                            'font_name': 'Calibri',
                            'border': 1,
                            'align': 'center',
                            'valign': 'vcenter',
                            'font_color': '#FF0000'
                        })
                    worksheet.write(row, col, value, cell_format)
                row += 1
            
            # 7. End of Report
            row += 1
            worksheet.merge_range(row, 0, row, 9, "End of Report", workbook.add_format({
                'italic': True,
                'font_name': 'Calibri',
                'align': 'center',
                'font_color': '#7F7F7F'
            }))

            # Save the Excel file
            writer.close()
            output.seek(0)

            # Create HTTP response
            response = HttpResponse(
                output.getvalue(),
                content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            filename = f"Salary_Statement_{employee_dict['Employee Name'].replace(' ', '_')}_{current_year}.xlsx"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

        except Exception as e:
            return HttpResponse(f"Error generating report: {str(e)}", status=500)

def report_view(request):
    transactions = Fund.objects.order_by('-id')[:10]
    # Corrected: Get distinct salary_year values directly
    salary_years = Salary.objects.values_list('salary_year', flat=True).distinct()
    # Fund years remain the same as date is a DateField
    fund_years = Fund.objects.dates('date', 'year').values_list('date__year', flat=True)
    all_years = sorted(set(salary_years) | set(fund_years), reverse=True)
    
    annual_data = []
    for year in all_years:
        fund_data = Fund.objects.filter(date__year=year)
        total_credits = fund_data.aggregate(Sum('credit'))['credit__sum'] or 0
        total_debits = fund_data.aggregate(Sum('debit'))['debit__sum'] or 0
        
        salary_data = Salary.objects.filter(salary_year=year)
        net_balance = salary_data.aggregate(Sum('net_salary'))['net_salary__sum'] or 0
        
        annual_data.append({
            'year': year,
            'total_credits': total_credits,
            'total_debits': total_debits,
            'net_balance': net_balance
        })
    
    context = {
        'annual_data': annual_data,
        'transactions': transactions
    }
    return render(request, 'report.html', context)

import os
import datetime
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from django.views import View
from django.http import HttpResponse
from django.utils import timezone
from django.db.models import Sum
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from .models import Fund, Salary, Department

logger = logging.getLogger(__name__)

@method_decorator(xframe_options_exempt, name='dispatch')
class AnnualFinancialReportPDF(View):
    def get(self, request, year, *args, **kwargs):
        try:
            buffer = BytesIO()
            
            # 1. Data Collection
            fund_data = self._get_fund_data(year)
            salary_data = self._get_salary_data(year)
            
            # 2. Generate Report with historical context
            response = self.generate_financial_report(fund_data, salary_data, buffer, year)
            
            # 3. Prepare Download Response
            response['Content-Type'] = 'application/pdf'
            response['Content-Disposition'] = f'attachment; filename="financial_report_{year}.pdf"'
            response['X-Content-Type-Options'] = 'nosniff'
            return response
            
        except Exception as e:
            logger.error(f"Financial report error ({year}): {str(e)}")
            return HttpResponse(f"Report generation failed: {str(e)}", status=500)

    def _get_fund_data(self, year):
        return Fund.objects.filter(date__year=year).order_by('date')

    def _get_salary_data(self, year):
        return Salary.objects.filter(salary_year=year).select_related('employee__department')

    def generate_financial_report(self, fund_data, salary_data, buffer, year):
        # ======================
        # 1. Data Processing
        # ======================
        # Current year data
        fund_df = pd.DataFrame(list(fund_data.values()))
        if not fund_df.empty:
            fund_df['date'] = pd.to_datetime(fund_df['date'])

        # Historical comparison data
        historical_salary = Salary.objects.filter(salary_year__lte=year)
        historical_df = pd.DataFrame(list(historical_salary.values(
            'salary_year', 
            'net_salary'
        )))

        # Current year department data
        dept_df = pd.DataFrame(list(salary_data.values(
            'employee__department__department_name',
            'net_salary'
        )))

        # ======================
        # 2. Chart Generation
        # ======================
        def create_financial_charts():
            chart_paths = {}
            sns.set_theme(style="whitegrid")  # Updated styling approach

            # Fund Trend Chart
            if not fund_df.empty:
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=fund_df['date'], y=fund_df['available_funds'], 
                            marker='o', color='#2c3e50')
                plt.title(f'Funds Flow - {year}', fontsize=14)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Amount ($)', fontsize=12)
                chart_paths['fund_trend'] = 'fund_trend.png'
                plt.savefig(chart_paths['fund_trend'], bbox_inches='tight')
                plt.close()

            # Department Salary Distribution
            if not dept_df.empty:
                plt.figure(figsize=(8, 8))
                dept_totals = dept_df.groupby('employee__department__department_name')['net_salary'].sum()
                colors = sns.color_palette("pastel")
                plt.pie(dept_totals, labels=dept_totals.index, colors=colors,
                       autopct='%1.1f%%', textprops={'fontsize': 12})
                plt.title(f'Salary Distribution - {year}', fontsize=14)
                chart_paths['dept_pie'] = 'dept_pie.png'
                plt.savefig(chart_paths['dept_pie'], bbox_inches='tight')
                plt.close()

            # Historical Comparison
            if not historical_df.empty:
                plt.figure(figsize=(10, 6))
                year_totals = historical_df.groupby('salary_year')['net_salary'].sum().sort_index()
                ax = sns.barplot(x=year_totals.index.astype(str), 
                                y=year_totals.values, 
                                palette="Blues_d")
                plt.title('Historical Salary Comparison', fontsize=14)
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Total Salary ($)', fontsize=12)
                ax.bar_label(ax.containers[0], fmt='$%.2f')
                chart_paths['year_comparison'] = 'year_comparison.png'
                plt.savefig(chart_paths['year_comparison'], bbox_inches='tight')
                plt.close()

            return chart_paths
        chart_paths = create_financial_charts()

        # ======================
        # 3. PDF Generation
        # ======================
        doc = SimpleDocTemplate(
            buffer,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        styles = self._create_styles()
        elements = self._create_report_elements(styles, chart_paths, fund_df, dept_df, historical_df, year)
        
        doc.build(elements)
        
        # Cleanup charts
        for chart in chart_paths.values():
            if os.path.exists(chart):
                os.remove(chart)
        
        return HttpResponse(buffer.getvalue(), content_type='application/pdf')

    def _create_styles(self):
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name='ReportTitle',
            fontName='Helvetica-Bold',
            fontSize=18,
            spaceAfter=0.5*inch,
            alignment=1
        ))
        styles.add(ParagraphStyle(
            name='SectionHeader',
            fontName='Helvetica-Bold',
            fontSize=14,
            spaceBefore=0.3*inch,
            spaceAfter=0.2*inch
        ))
        return styles

    def _create_report_elements(self, styles, chart_paths, fund_df, dept_df, historical_df, year):
        elements = []
        import datetime
        # Cover Page
        elements.append(Paragraph(f"{year} Financial Report", styles['ReportTitle']))
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(f"Generated on: {datetime.date.today().strftime('%B %d, %Y')}", styles['Normal']))
        elements.append(PageBreak())
        
        # Bank Transactions
        if not fund_df.empty:
            elements.append(Paragraph("1. Fund Transactions", styles['SectionHeader']))
            elements.append(self._create_fund_table(fund_df))
            if 'fund_trend' in chart_paths:
                elements.append(Spacer(1, 0.2*inch))
                elements.append(Image(chart_paths['fund_trend'], width=6*inch, height=3*inch))
            elements.append(PageBreak())
        
        # Department Salaries
        if not dept_df.empty:
            elements.append(Paragraph("2. Department Breakdown", styles['SectionHeader']))
            elements.append(self._create_dept_table(dept_df))
            if 'dept_pie' in chart_paths:
                elements.append(Spacer(1, 0.2*inch))
                elements.append(Image(chart_paths['dept_pie'], width=5*inch, height=5*inch))
            elements.append(PageBreak())
        
        # Historical Comparison
        if not historical_df.empty:
            elements.append(Paragraph("3. Historical Comparison", styles['SectionHeader']))
            if 'year_comparison' in chart_paths:
                elements.append(Image(chart_paths['year_comparison'], width=6*inch, height=4*inch))
            elements.append(PageBreak())
        
        # Financial Summary
        elements.append(Paragraph("4. Financial Summary", styles['SectionHeader']))
        elements.append(self._create_summary_table(fund_df, dept_df, year))
        
        return elements

    def _create_fund_table(self, df):
        headers = ["Date", "Credit", "Debit", "Balance"]
        data = [headers]
        for _, row in df.iterrows():
            data.append([
                row['date'].strftime('%b %d'),
                f"${row['credit']:,.2f}",
                f"${row['debit']:,.2f}",
                f"${row['available_funds']:,.2f}"
            ])
        
        return Table(data, style=[
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4B77BE')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
        ], colWidths=[1.5*inch]*4)

    def _create_dept_table(self, df):
        dept_totals = df.groupby('employee__department__department_name')['net_salary'].sum()
        data = [["Department", "Total Salary"]]
        for dept, total in dept_totals.items():
            data.append([dept, f"${total:,.2f}"])
        
        return Table(data, style=[
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3FC380')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTSIZE', (0,0), (-1,0), 10),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
        ], colWidths=[3*inch, 2*inch])

    def _create_summary_table(self, fund_df, salary_df, year):
        summary_data = []
        
        # Fund Summary
        if not fund_df.empty:
            summary_data.extend([
                ["Total Credits", f"${fund_df['credit'].sum():,.2f}"],
                ["Total Debits", f"${fund_df['debit'].sum():,.2f}"],
                ["Ending Balance", f"${fund_df['available_funds'].iloc[-1]:,.2f}"]
            ])
        
        # Salary Summary
        if not salary_df.empty:
            total_salary = salary_df['net_salary'].sum()
            avg_salary = salary_df['net_salary'].mean()
            summary_data.extend([
                ["Total Salaries", f"${total_salary:,.2f}"],
                ["Average Salary", f"${avg_salary:,.2f}"],
                ["Total Employees", salary_df['employee__department__department_name'].nunique()]
            ])
        
        return Table([["Financial Summary - {}".format(year)]] + summary_data, style=[
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F2784B')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('SPAN', (0,0), (1,0)),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
        ], colWidths=[3*inch, 2*inch])
    
# views.py
from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from model.illama_resume_details import process_resume
from model.illama_resume_weakness import process_weakness_resume
import os
import threading
import uuid
from datetime import datetime
from django.core.cache import cache
def upload_resume(request):
    if request.method == 'POST':
        resume_file = request.FILES.get('resume')

        # Validate file
        if not resume_file:
            messages.error(request, 'Please select a file first!')
            return redirect('upload_resume')

        if resume_file.size > 5 * 1024 * 1024:
            messages.error(request, 'File size exceeds 5MB limit!')
            return redirect('upload_resume')

        valid_extensions = ['.pdf', '.doc', '.docx']
        if not any(resume_file.name.lower().endswith(ext) for ext in valid_extensions):
            messages.error(request, 'Invalid file type! Allowed types: PDF, DOC, DOCX')
            return redirect('upload_resume')

        processing_id = str(uuid.uuid4())
        cache.set(f'processing_{processing_id}', False, timeout=300)  # 5 minute timeout

        fs = FileSystemStorage(location='media/')
        os.makedirs('media', exist_ok=True)
        filename = fs.save(resume_file.name, resume_file)
        file_path = os.path.join('media', filename)

        def background_task(path, proc_id):
            try:
                process_weakness_resume(path)
                # Mark processing as complete
                cache.set(f'processing_{proc_id}', True)
            finally:
                if os.path.exists(path):
                    os.remove(path)

        threading.Thread(target=background_task, args=(file_path, processing_id)).start()
        
        # Process resume using your parser
        data = process_resume(file_path)
        if not data:
            messages.error(request, 'Resume processing failed.')
            return redirect('upload_resume')

        return render(request, 'resume_details.html', {
            'data': data,
            'processing_id': processing_id
        })
    return render(request, 'upload.html')
def check_processing(request):
    processing_id = request.GET.get('id')
    is_complete = cache.get(f'processing_{processing_id}', False)
    return JsonResponse({'complete': is_complete})
def chatbot_view(request):
    return render(request, 'chatbot.html')

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from pathlib import Path
from model.hr_chatbot import LlamaNegotiationExpert, SmartHRInterface, NegotiationState  # Make sure to import NegotiationState

# Global session variables
negotiation_session = None
position_selected = False

import os
from pathlib import Path

def generate_llama_response(user_message=""):
    global negotiation_session, position_selected
    
    # Define paths
    MEDIA_PATH = r"C:\Users\athis\OneDrive\Documents\PSG\Project\Payroll System\app\media"
    REPORT_PATH = os.path.join(MEDIA_PATH, "resume.json")
    
    # Initial message when the page is loaded
    if not user_message:
        try:
            # Reset session state
            negotiation_session = None
            position_selected = False
            
            # Initialize the negotiation system
            expert = LlamaNegotiationExpert()
            
            # Check if report exists in media directory
            if not Path(REPORT_PATH).exists():
                return "Error: Resume analysis not found. Please upload your resume again."
            
            # Load resume data to get email
            with open(REPORT_PATH, 'r') as f:
                resume_data = json.load(f)
            candidate_email = resume_data.get('structured_data', {}).get('candidate_overview', {}).get('email', '')
            
            negotiation_session = SmartHRInterface(REPORT_PATH)
            negotiation_session.candidate_email = candidate_email  # Store email for later use
            
            roles = negotiation_session._select_role()
            
            response = "Based on your profile, you are suitable for:"
            for idx, role in enumerate(roles, 1):
                response += f"{idx}. {role}\n"
            response += "\nPlease reply with the number of the position you'd like to discuss."
            
            return response
        except Exception as e:
            return f"System error: {str(e)}"
    
    # For follow-up queries
    try:
        # Check if user is selecting a position
        if negotiation_session and not position_selected:
            if user_message.isdigit():
                roles = negotiation_session._select_role()
                idx = int(user_message) - 1
                if 0 <= idx < len(roles):
                    selected_role = roles[idx]
                    negotiation_session.temp_func(selected_role)
                    position_selected = True
                    
                    initial_details = negotiation_session._display_offer_details()
                    response = f"HR: Welcome {initial_details['candidate_name']}!\n"
                    response += f"Position: {initial_details['position']}\n"
                    response += f"Initial Offer: {initial_details['initial_offer']}\n"
                    response += f"Market Range: {initial_details['market_range']}\n"
                    response += "Development Benefits:\n"
                    for benefit in initial_details['development_benefits']:
                        response += f"- {benefit.get('type', 'Benefit')}: {benefit.get('recommendation', '')}\n"
                    response += initial_details['message']
                    return response
                else:
                    return "Invalid selection. Please choose a number from the list."
            else:
                return "Please select a position by number to begin negotiation."
        
        # Normal negotiation flow
        if negotiation_session and position_selected:
            if not hasattr(negotiation_session, 'engine'):
                return "Negotiation system error. Please start a new session by refreshing the page."
            
            current_offer, hr_response = negotiation_session.engine.generate_counteroffer(user_message)
            
            # Clean up HR response
            if hr_response.startswith("We're pleased to offer"):
                hr_response = hr_response.split('"', 1)[-1]
                
            closing_phrases = [
                "Thank you for your prompt response",
                "Best regards,",
                "[HR Representative]",
                "[Your Name] HR Representative",
                "Here's a potential response from the HR:"
                " Here's the HR's potential response: ",
                "Dear Candidate, "
            ]
            
            for phrase in closing_phrases:
                hr_response = hr_response.replace(phrase, "")
            
            # Check if negotiation is closed
            if negotiation_session.engine.state == NegotiationState.CLOSED:
                status = 'accepted' if 'congratulations' in hr_response.lower() else 'declined'
                final_offer = negotiation_session.engine.current_offer
                offered_benefits = getattr(negotiation_session.engine, 'offered_benefits', {})
                
                # Prepare negotiation result
                response = f"HR: {hr_response}\n\n=== Negotiation Result ===\n"
                response += f"Final Offer: {final_offer} LPA\n"
                response += f"Status: {status.upper()}\n"
                if offered_benefits:
                    response += "Agreed Benefits:\n"
                    for category, requests in offered_benefits.items():
                        response += f"- {category.title()}: {', '.join(requests[:2])}\n"
                if status == 'accepted':
                    response += "\nNext Steps: HR will contact you within 24 hours for onboarding."
                else:
                    response += "\nThank you for your time. We wish you the best in your future endeavors."
                
                # Send email with negotiation details
                try:
                    if hasattr(negotiation_session, 'candidate_email') and negotiation_session.candidate_email:
                        send_negotiation_email(
                            negotiation_session.candidate_email,
                            position=negotiation_session.engine.role,
                            final_offer=final_offer,
                            status=status,
                            offered_benefits=offered_benefits
                        )
                except Exception as email_error:
                    print(f"Error sending email: {email_error}")
                
                # Cleanup: Remove resume files
                try:
                    if os.path.exists(REPORT_PATH):
                        os.remove(REPORT_PATH)
                        print(f"Removed resume file: {REPORT_PATH}")
                    
                    # Remove other temporary files
                    for file in os.listdir(MEDIA_PATH):
                        if file.endswith(('.json', '.pdf', '.docx')):
                            file_path = os.path.join(MEDIA_PATH, file)
                            os.remove(file_path)
                            print(f"Removed temporary file: {file_path}")
                except Exception as cleanup_error:
                    print(f"Error during cleanup: {cleanup_error}")
                
                # Reset session
                negotiation_session = None
                position_selected = False
                return response
            
            return f"HR: {hr_response}\nCurrent Offer: {current_offer}"
        
        return "Please start a new negotiation session by refreshing the page."
    
    except Exception as e:
        # Attempt cleanup even on error
        try:
            if os.path.exists(REPORT_PATH):
                os.remove(REPORT_PATH)
        except:
            pass
        
        negotiation_session = None
        position_selected = False
        return f"An error occurred: {str(e)}. Session has been reset."

def send_negotiation_email(email, position, final_offer, status, offered_benefits):
    """Send email with negotiation details to candidate"""
    from django.core.mail import send_mail
    from django.conf import settings
    
    subject = f"Your Negotiation Results - {position}"
    
    body = f"""
Dear Candidate,

We're writing to confirm the details of your recent negotiation:

Position: {position}
Final Offer: {final_offer} LPA
Status: {status.upper()}

{"Next Steps: HR will contact you within 24 hours for onboarding." if status == 'accepted' else "Thank you for your time and consideration."}

Best regards,
HR Team
"""
    
    send_mail(
        subject=subject,
        message=body.strip(),
        from_email=settings.DEFAULT_FROM_EMAIL,
        recipient_list=[email],
        fail_silently=False,
    )
@csrf_exempt
def chatbot_reply(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")
            
            # Get the response
            reply = generate_llama_response(user_message)
            
            return JsonResponse({"reply": reply})
        except json.JSONDecodeError:
            return JsonResponse({"reply": "Invalid request format"}, status=400)
    return JsonResponse({"reply": "Method not allowed"}, status=405)