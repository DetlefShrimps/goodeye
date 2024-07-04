document.addEventListener('DOMContentLoaded', function () {
    const pitcherSearch = document.getElementById('pitcher-search');
    const batterSearch = document.getElementById('batter-search');
    const playBallButton = document.getElementById('play-ball-button');
    const clearButton = document.getElementById('clear-button');
    const predictionElement = document.getElementById('prediction');

    pitcherSearch.addEventListener('input', function () {
        const query = pitcherSearch.value;
        fetch(`/search_player?q=${query}`)
            .then(response => response.json())
            .then(data => {
                // Populate search suggestions for pitchers
                console.log(data);
            });
    });

    batterSearch.addEventListener('input', function () {
        const query = batterSearch.value;
        fetch(`/search_player?q=${query}`)
            .then(response => response.json())
            .then(data => {
                // Populate search suggestions for batters
                console.log(data);
            });
    });

    playBallButton.addEventListener('click', function () {
        const pitcher = { /* get pitcher data from selection */ };
        const batter = { /* get batter data from selection */ };

        fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pitcher, batter })
        })
        .then(response => response.json())
        .then(data => {
            predictionElement.textContent = data.prediction;
        });
    });

    clearButton.addEventListener('click', function () {
        pitcherSearch.value = '';
        batterSearch.value = '';
        predictionElement.textContent = '';
    });
});
