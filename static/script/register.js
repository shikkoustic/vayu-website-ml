function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

document.getElementById('registerForm').addEventListener('submit', function(event) {
  const email = document.getElementById('email').value;
  const password = document.getElementById('password').value;
  const emailErrorElement = document.getElementById('emailError');
  const passwordErrorElement = document.getElementById('passwordError');
  
  let isValid = true;
  emailErrorElement.textContent = '';
  passwordErrorElement.textContent = '';
  

  if (!validateEmail(email)) {
    event.preventDefault();
    emailErrorElement.textContent = 'Please enter a valid email address (e.g., abcd@yourdomain.com)';
    isValid = false;
  }
  
  if (password.length < 6) {
    event.preventDefault();
    passwordErrorElement.textContent = 'Password must be at least 6 characters long';
    isValid = false;
  }
  
  if (!/\d/.test(password)) {
    event.preventDefault();
    passwordErrorElement.textContent = 'Password must contain at least one number (0-9)';
    isValid = false;
  }
  
  return isValid;
});

document.getElementById('email').addEventListener('blur', function() {
  const email = this.value;
  const errorElement = document.getElementById('emailError');
  
  if (email.length > 0 && !validateEmail(email)) {
    errorElement.textContent = 'Please enter a valid email address (e.g., abcd@yourdomain.com)';
  } else {
    errorElement.textContent = '';
  }
});

document.getElementById('password').addEventListener('input', function() {
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