document.addEventListener("DOMContentLoaded", function () {
    const historyBtn = document.getElementById("historyBtn");
    const historyPanel = document.getElementById("historyPanel");
    const historyList = document.getElementById("historyList");
    const predictionForm = document.getElementById("predictionForm");
    const result = document.getElementById("result");
    
    let historyData = JSON.parse(localStorage.getItem("predictionHistory")) || [];
    
    function updateHistoryUI() {
        historyList.innerHTML = "";
        historyData.forEach(entry => {
            let li = document.createElement("li");
            li.textContent = `${entry.date} - ${entry.prediction}`;
            historyList.appendChild(li);
        });
    }
    
    historyBtn.addEventListener("click", function () {
        historyPanel.classList.toggle("show");
        updateHistoryUI();
    });
    
    predictionForm.addEventListener("submit", function (event) {
        event.preventDefault();
        
        const price = document.getElementById("price").value;
        const discount = document.getElementById("discount").value;
        const shipping = document.getElementById("shipping").value;
        
        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ price, discount, shipping })
        })
        .then(response => response.json())
        .then(data => {
            let predictionText = `Prediction: ${data.prediction}`;
            if (data.prediction.toLowerCase() !== "returned") {
                predictionText += ` (${data.probability} probability)`;
            }
            result.textContent = predictionText;
            
            let newEntry = {
                date: new Date().toLocaleString(),
                prediction: predictionText
            };
            historyData.push(newEntry);
            localStorage.setItem("predictionHistory", JSON.stringify(historyData));
        })
        .catch(error => console.error("Error:", error));
    });
});
