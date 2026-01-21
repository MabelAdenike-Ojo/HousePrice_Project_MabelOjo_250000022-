const form = document.getElementById("predictForm");

form.addEventListener("submit", async function(e) {
    e.preventDefault(); // prevent page reload

    const data = {
        GrLivArea: Number(document.getElementById("GrLivArea").value),
        BedroomAbvGr: Number(document.getElementById("BedroomAbvGr").value),
        Neighborhood: document.getElementById("Neighborhood").value
    };

    console.log("Sending data to API:", data); // debug

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        console.log("API response:", result); // debug

        if (result.Predicted_House_Price !== undefined) {
            document.getElementById("result").innerText =
                "Predicted Price: $" + result.Predicted_House_Price;
        } else {
            document.getElementById("result").innerText =
                "Error: " + (result.error || "Unknown error");
        }

    } catch (error) {
        console.error("Fetch error:", error);
        document.getElementById("result").innerText = "Error connecting to server";
    }
});