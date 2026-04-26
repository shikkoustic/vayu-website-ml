document.getElementById('resetPasswordForm').addEventListener('submit', function(event) {
  const newPassword = document.getElementById('new_password').value;
  const confirmPassword = document.getElementById('confirm_password').value;
  const passwordErrorElement = document.getElementById('passwordError');
  const confirmPasswordErrorElement = document.getElementById('confirmPasswordError');
  
  let isValid = true;
  
  passwordErrorElement.textContent = '';
  confirmPasswordErrorElement.textContent = '';
  
  if (newPassword.length < 6) {
    event.preventDefault();
    passwordErrorElement.textContent = 'Password must be at least 6 characters long';
    isValid = false;
  }
  
  if (!/\d/.test(newPassword)) {
    event.preventDefault();
    passwordErrorElement.textContent = 'Password must contain at least one number (0-9)';
    isValid = false;
  }
  
  if (newPassword !== confirmPassword) {
    event.preventDefault();
    confirmPasswordErrorElement.textContent = 'Passwords do not match';
    isValid = false;
  }
  
  return isValid;
});

document.getElementById('new_password').addEventListener('input', function() {
  const password = this.value;
  const errorElement = document.getElementById('passwordError');
  
  if (password.length > 0) {
    if (password.length < 6) {
      errorElement.textContent = 'Password must be at least 6 characters long';
    } else if (!/\d/.test(password)) {
      errorElement.textContent = 'Password must contain at least one number (0-9)';
    } else {
      errorElement.textContent = '';
    }
  } else {
    errorElement.textContent = '';
  }
});

document.getElementById('confirm_password').addEventListener('input', function() {
  const newPassword = document.getElementById('new_password').value;
  const confirmPassword = this.value;
  const errorElement = document.getElementById('confirmPasswordError');
  
  if (confirmPassword.length > 0) {
    if (newPassword !== confirmPassword) {
      errorElement.textContent = 'Passwords do not match';
    } else {
      errorElement.textContent = '';
    }
  } else {
    errorElement.textContent = '';
  }
});