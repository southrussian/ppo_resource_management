document.addEventListener('DOMContentLoaded', function() {
    const numPatientsInput = document.getElementById('num_patients');
    const patientFields = document.getElementById('patient-fields');
    const targetStateFields = document.getElementById('target-state-fields');

    numPatientsInput.addEventListener('input', function() {
        const numPatients = parseInt(numPatientsInput.value);
        patientFields.innerHTML = '';
        for (let i = 0; i < numPatients; i++) {
            const patientDiv = document.createElement('div');
            patientDiv.classList.add('form-group');
            patientDiv.innerHTML = `
                <label for="urgency_${i}">Urgency for Patient ${i+1} (1-3):</label>
                <input type="number" id="urgency_${i}" name="urgency_${i}" min="1" max="3" required>
                <label for="completeness_${i}">Completeness for Patient ${i+1} (0-1):</label>
                <input type="number" id="completeness_${i}" name="completeness_${i}" min="0" max="1" required>
                <label for="complexity_${i}">Complexity for Patient ${i+1} (0-1):</label>
                <input type="number" id="complexity_${i}" name="complexity_${i}" min="0" max="1" required>
                <label for="initial_position_${i}">Initial Position for Patient ${i+1} (0-6):</label>
                <input type="number" id="initial_position_${i}" name="initial_position_${i}" min="0" max="6" required>
            `;
            patientFields.appendChild(patientDiv);
        }
    });

    for (let i = 0; i < 7; i++) {
        const targetStateDiv = document.createElement('div');
        targetStateDiv.classList.add('form-group');
        targetStateDiv.innerHTML = `
            <label for="min_${i}">Min Target Value for Day ${i}:</label>
            <input type="number" id="min_${i}" name="min_${i}" required>
            <label for="max_${i}">Max Target Value for Day ${i}:</label>
            <input type="number" id="max_${i}" name="max_${i}" required>
        `;
        targetStateFields.appendChild(targetStateDiv);
    }
});
