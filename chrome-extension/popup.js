const apiUrl = ""; // Replace with your API URL

// Function to handle the "Predict" button click event
function handlePredictButtonClick() {
  const fileInput = document.getElementById("file-upload");
  const file = fileInput.files[0];

  // Create a FormData object to send the file to the API
  const formData = new FormData();
  formData.append("image", file);

  // Send a POST request to the API with the file data
  fetch(apiUrl, {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      // Handle the response from the API
      const captchaText = data.captcha_text;
      alert(`The predicted captcha text is: ${captchaText}`);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

// Add an event listener to the "Predict" button
const predictButton = document.getElementById("predict-button");
predictButton.addEventListener("click", handlePredictButtonClick);
