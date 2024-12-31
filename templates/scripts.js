$(document).ready(function () {
    function validateInputs() {
        let isValid = true;
        $('.error').text('');

        const x0 = $('#x0').val();
        const y0 = $('#y0').val();
        const step_value = $('#step_value').val();
        const equation = $('#equation').val();

        if (x0 || y0 || step_value || equation) {
            if (!x0) {
                $('#x0-error').text('x0 is required.');
                isValid = false;
            }
            if (!y0) {
                $('#y0-error').text('xf is required.');
                isValid = false;
            }
            if (!step_value) {
                $('#step_value-error').text('Number of steps is required.');
                isValid = false;
            }
            if (!equation) {
                $('#equation-error').text('Equation is required.');
                isValid = false;
            }
        }

        return isValid;
    }

    function generatePlot() {
        if (validateInputs()) {
            $.ajax({
                url: '/',
                type: 'POST',
                data: $('#parameters-form').serialize() + '&' + $('#points-form').serialize() + '&' + $('#algorithms-form').serialize(),
                success: function (response) {
                    $('#plot-div').html(response.plot_html);
                    $('#parameters-form-error').text('');
                    $('#points-form-error').text('');
                    $('#algorithms-form-error').text('');
                    populatePointsTable(response.points);
                },
                error: function (xhr, status, error) {
                    $('#parameters-form-error').text('An error occurred: ' + xhr.responseText);
                    $('#points-form-error').text('An error occurred: ' + xhr.responseText);
                    $('#algorithms-form-error').text('An error occurred: ' + xhr.responseText);
                }
            });
        }
    }

    function populatePointsTable(points) {
        $('#points-tbody').empty();
        points.forEach(point => {
            $('#points-tbody').append(`<tr>
                <td>${point.x}</td>
                <td>${point.y}</td>
                <td>${point.cubic_spline || ''}</td>
                <td>${point.cubic_spline_error || ''}</td>
                <td>${point.newton || ''}</td>
                <td>${point.newton_error || ''}</td>
                <td>${point.vandermonde || ''}</td>
                <td>${point.vandermonde_error || ''}</td>
            </tr>`);
        });
    }

    function sortTable() {
        let rows = $('#points-input-table tbody tr').get();
        rows.sort(function (a, b) {
            let A = parseFloat($(a).find('input[name="x_values[]"]').val());
            let B = parseFloat($(b).find('input[name="x_values[]"]').val());
            return A - B;
        });
        $.each(rows, function (index, row) {
            $('#points-input-table tbody').append(row);
        });
    }

    $('#parameters-form').on('submit', function (event) {
        event.preventDefault();
        generatePlot();
    });

    $('#points-form').on('submit', function (event) {
        event.preventDefault();
        generatePlot();
    });

    $('#algorithms-form').on('submit', function (event) {
        event.preventDefault();
        generatePlot();
    });

    $('#x0, #y0, #step_value, #equation').on('input', function () {
        generatePlot();
    });

    $('#cubic_spline, #newton, #vandermonde').on('change', function () {
        generatePlot();
    });

    $('#points-input-table').on('focusout', 'input', function () {
        sortTable();
    });

    $('#generate-plot').on('click', function () {
        generatePlot();
    });

    $('#add-point').on('click', function () {
        $('#points-input-table tbody').append(`<tr>
            <td><input type="number" class="form-control table-input" name="x_values[]"></td>
            <td><input type="number" class="form-control table-input" name="y_values[]"></td>
            <td><button type="button" class="btn btn-sm btn-danger remove-point">-</button></td>
        </tr>`);
    });

    $(document).on('click', '.remove-point', function () {
        $(this).closest('tr').remove();
    });

    $('.collapsible-header').on('click', function () {
        $(this).find('span').toggleClass('rotate');
        $(this).next('.collapsible-content').toggleClass('show');
    });
});
