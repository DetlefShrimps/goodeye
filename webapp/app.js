document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('pitcher').addEventListener('change', function () {
        fetchYears('pitcher', 'pitcher_year');
    });

    document.getElementById('batter').addEventListener('change', function () {
        fetchYears('batter', 'batter_year');
    });

    document.getElementById('prediction-form').addEventListener('submit', function (event) {
        event.preventDefault();
        predictOutcome();
    });
});

function fetchYears(playerType, yearDropdownId) {
    const playerName = document.getElementById(playerType).value;
    const yearDropdown = document.getElementById(yearDropdownId);

    fetch('/get_years', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `player_name=${playerName}`
    })
        .then(response => response.json())
        .then(data => {
            yearDropdown.innerHTML = '';
            data.forEach(year => {
                const option = document.createElement('option');
                option.value = year;
                option.textContent = year;
                yearDropdown.appendChild(option);
            });
        });
}

function predictOutcome() {
    const formData = new FormData(document.getElementById('prediction-form'));

    fetch('/predict', {
        method: 'POST',
        body: new URLSearchParams(formData)
    })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            if (data.error) {
                resultDiv.textContent = `Error: ${data.error}`;
            } else {
                resultDiv.textContent = `Prediction result: ${data.result}`;
            }
        });
}
