from django import forms
from .models import Employee,Department,Designation,LeaveRecord

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150, widget=forms.TextInput(attrs={
        'class': 'input-field',
        'placeholder': 'Username'
    }))
    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'class': 'input-field',
        'placeholder': 'Password'
    }))

class ForgotPasswordForm(forms.Form):
    employee_id = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'input-field',
            'placeholder': 'Employee ID'
        })
    )
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'input-field',
            'placeholder': 'Email'
        })
    )


class EmployeeForm(forms.ModelForm):
    department = forms.ModelChoiceField(queryset=Department.objects.all(), required=True)
    designation = forms.ModelChoiceField(queryset=Designation.objects.all(), required=True)  # Use ModelChoiceField for Designation
    class Meta:
        model = Employee
        fields = [
            'first_name', 'last_name', 'date_of_birth', 'gender',
            'email', 'contact_info', 'address', 'joining_date',
            'department','designation', 'bank_account_no', 'pan_no', 'status'
        ]

        widgets = {
            'date_of_birth': forms.DateInput(attrs={'type': 'date'}),
            'joining_date': forms.DateInput(attrs={'type': 'date'}),
            'gender': forms.Select(choices=[
                ('', 'Select Gender'),
                ('Male', 'Male'),
                ('Female', 'Female'),
                ('Other', 'Other')
            ]),
            'status': forms.Select(choices=[
                ('', 'Select Status'),
                ('active', 'Active'),
                ('terminated', 'Terminated')
            ]),
            'address': forms.Textarea(attrs={'rows': 3}),
        }
        labels = {
            'designation': 'Designation',
            'department': 'Department',
            'bank_account_no': 'Bank Account Number',
            'pan_no': 'PAN Number',
        }

class LeaveRequestForm(forms.ModelForm):
    class Meta:
        model = LeaveRecord
        fields = ['leave_type', 'start_date', 'end_date', 'reason']
        widgets = {
            'leave_type': forms.Select(attrs={'class': 'form-select'}),
            'start_date': forms.DateInput(attrs={'type': 'date'}),
            'end_date': forms.DateInput(attrs={'type': 'date'}),
            'reason': forms.Textarea(attrs={'rows': 3}),
        }


class SalaryPredictionForm(forms.Form):
    experience = forms.IntegerField(min_value=0, label="Years of Experience")
    designation = forms.CharField(label="Designation")
    education = forms.ChoiceField(label="Education")

    def __init__(self, *args, **kwargs):
        designations = kwargs.pop('designations', [])
        educations = kwargs.pop('educations', [])
        super(SalaryPredictionForm, self).__init__(*args, **kwargs)
        self.fields['designation'].widget = forms.Select(choices=[(d, d) for d in designations])
        self.fields['education'].choices = [(e, e) for e in educations]

# forms.py
from django import forms
from .models import TaxInput

from django import forms
from .models import TaxInput

class TaxInputForm(forms.ModelForm):
    class Meta:
        model = TaxInput
        exclude = ['employee', 'total_tax', 'created_at']
        widgets = {
            'rent_paid': forms.NumberInput(attrs={'class': 'form-control'}),
            'lic': forms.NumberInput(attrs={'class': 'form-control'}),
            'epf': forms.NumberInput(attrs={'class': 'form-control'}),
            'home_principal': forms.NumberInput(attrs={'class': 'form-control'}),
            'mutual_fund': forms.NumberInput(attrs={'class': 'form-control'}),
            'nps_contribution': forms.NumberInput(attrs={'class': 'form-control'}),
            'health_insurance': forms.NumberInput(attrs={'class': 'form-control'}),
            'parent_health': forms.NumberInput(attrs={'class': 'form-control'}),
            'home_loan_interest': forms.NumberInput(attrs={'class': 'form-control'}),
            'education_expense': forms.NumberInput(attrs={'class': 'form-control'}),
            'donations': forms.NumberInput(attrs={'class': 'form-control'}),
        }
