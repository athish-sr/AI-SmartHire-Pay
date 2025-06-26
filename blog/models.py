from django.db import models

from django.db import models

class Department(models.Model):
    department_id = models.AutoField(primary_key=True)
    department_name = models.CharField(max_length=100, unique=True)

    class Meta:
        db_table = 'departments'
        managed=False
    
    def __str__(self):
        return self.department_name

class Designation(models.Model):
    designation_id = models.AutoField(primary_key=True)
    designation_title = models.CharField(max_length=100)
    basic_salary = models.DecimalField(max_digits=12, decimal_places=2)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)

    class Meta:
        db_table = 'designations'
        managed=False
    def __str__(self):
        return self.designation_title

class Employee(models.Model):
    employee_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    date_of_birth = models.DateField()
    gender = models.CharField(max_length=10)
    email = models.EmailField(unique=True)
    contact_info = models.CharField(max_length=100, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    joining_date = models.DateField()
    department = models.ForeignKey(Department, null=True, blank=True, on_delete=models.SET_NULL)
    designation = models.ForeignKey(Designation, null=True, blank=True, on_delete=models.SET_NULL)
    bank_account_no = models.CharField(max_length=30, null=True, blank=True)
    pan_no = models.CharField(max_length=20, null=True, blank=True)
    status = models.CharField(max_length=20)

    class Meta:
        db_table = 'employees'
        managed=False

class User(models.Model):
    user_id = models.AutoField(primary_key=True)
    username = models.CharField(max_length=50, unique=True)
    password_hash = models.TextField()
    role = models.CharField(max_length=50)
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE, db_column='employee_id', unique=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = 'users'
        managed=False

from django.db import models
from django.utils import timezone
from datetime import timedelta
from .models import User  # Adjust import if User is in another module
import datetime
class PasswordResetToken(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    def is_valid(self):
        return timezone.now() < self.expires_at

class LeaveRecord(models.Model):
    LEAVE_CHOICES = [
        ('Casual Leave', 'Casual Leave'),
        ('Sick Leave', 'Sick Leave'),
        ('Privilege Leave', 'Privilege Leave'),
        ('Maternity Leave', 'Maternity Leave'),
        ('Unpaid Leave', 'Unpaid Leave'),
    ]
    STATUS_CHOICES = [
        ('Pending', 'Pending'),
        ('Accepted', 'Accepted'),
        ('Rejected', 'Rejected'),
    ]

    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    leave_type = models.CharField(max_length=50, choices=LEAVE_CHOICES)
    start_date = models.DateField()
    end_date = models.DateField()
    reason = models.TextField(null=True, blank=True)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default='Pending')
    applied_on = models.DateTimeField(auto_now_add=True)
    approved_on = models.DateTimeField(null=True, blank=True)
    approved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)  # Who approved the leave

    def __str__(self):
        return f"{self.employee.first_name} {self.employee.last_name} - {self.leave_type}"
    
    class Meta:
        db_table = 'leave_record'
        managed=False

from django.db import models

class Attendance(models.Model):
    STATUS_CHOICES = [
        ('Present', 'Present'),
        ('Absent', 'Absent'),
        ('Leave', 'Leave'),
        ('Half-Day', 'Half-Day'),
        ('Work From Home', 'Work From Home'),
    ]

    attendance_id = models.AutoField(primary_key=True)
    employee = models.ForeignKey('Employee', on_delete=models.CASCADE, db_column='employee_id')
    date = models.DateField()
    status = models.CharField(max_length=20, choices=STATUS_CHOICES)
    check_in_time = models.TimeField(null=True, blank=True)
    check_out_time = models.TimeField(null=True, blank=True)

    class Meta:
        db_table = 'attendance'  # Use existing table
        managed = False          # Do not let Django manage this table
        unique_together = ('employee', 'date')
        ordering = ['-date']

    def __str__(self):
        return f"{self.employee_id} - {self.date} - {self.status}"

from django.db import models
from django.db import models
from .models import Employee

class TaxInput(models.Model):
    employee = models.OneToOneField(Employee, on_delete=models.CASCADE, primary_key=True)

    rent_paid = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    lic = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    epf = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    home_principal = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    mutual_fund = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    nps_contribution = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    health_insurance = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    parent_health = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    home_loan_interest = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    education_expense = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    donations = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    total_tax = models.DecimalField(max_digits=12, decimal_places=2, default=0)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"TaxInput for {self.employee.first_name} {self.employee.last_name}"
    class Meta:
        db_table = 'tax_details'  # Use existing table
        managed = False 


from django.db import models

class Salary(models.Model):
    PAYMENT_STATUS_CHOICES = [
        ('paid', 'Paid'),
        ('unpaid', 'Unpaid'),
    ]

    salary_id = models.AutoField(primary_key=True)
    employee = models.ForeignKey('Employee', on_delete=models.CASCADE, related_name='salaries')
    salary_month = models.IntegerField()  # <-- Add this
    salary_year = models.IntegerField()   # <-- Add this
    basic_salary = models.DecimalField(max_digits=12, decimal_places=2)
    allowances = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    deductions = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    bonus = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    net_salary = models.DecimalField(max_digits=12, decimal_places=2)
    payment_status = models.CharField(max_length=20, choices=PAYMENT_STATUS_CHOICES)
    tax = models.DecimalField(max_digits=12, decimal_places=2)

    def __str__(self):
        return f"{self.employee} - {self.pay_date} - {self.payment_status.capitalize()}"

    class Meta:
        db_table = 'salaries'
        managed = False

class Payslip(models.Model):
    payslip_id = models.AutoField(primary_key=True)
    employee = models.ForeignKey(Employee, on_delete=models.CASCADE)
    month = models.IntegerField()
    year = models.IntegerField()
    generated_on = models.DateTimeField(auto_now_add=True)
    generated_by = models.CharField(max_length=100)
    pdf_file = models.BinaryField()

    class Meta:
        db_table = 'payslips'
        managed = False


class notifications(models.Model):
    CATEGORY_CHOICES = (
        ('salary', 'Salary'),
        ('leave', 'Leave'),
        ('tax', 'Tax'),
    )
    
    employee = models.ForeignKey('Employee', on_delete=models.CASCADE)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return self.message
    
    class Meta:
        db_table = 'notifications'
        managed = False

from django.db import models
from django.utils import timezone

from django.db import models

class Fund(models.Model):
    date = models.DateField(auto_now_add=True)  # Defaults to current date
    credit = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    debit = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    available_funds = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    
    class Meta:
        db_table = 'fund'  # This ensures that the table is named 'fund' in the database.
        managed = False
