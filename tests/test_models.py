import pytest
import numpy.testing as npt
from ftir.models import calculate_difference_absorbance

@pytest.mark.parametrize('test_sample_intensities, test_reference_intensities, expected_output, expected_raises, error_message',
[
    ([[0.0345, 0.0348, 0.0329, 0.0398, 0.0483]],
    [[0.0322, 0.0319, 0.0299, 0.0374, 0.0458]],
    [[-0.068993, -0.087011, -0.095614, -0.062196, -0.053147]],
    None,
    None)
],
ids=['valid inputs'])
def test_difference_absorbance(test_sample_intensities, test_reference_intensities, expected_output, expected_raises, error_message):
    if expected_raises is None:
        result_data = difference_absorbance(test_sample_intensities, test_reference_intensities)
        npt.assert_array_almost_equal(result_data, expected_output)
    else:
        with pytest.raises(expected_raises, match=error_message):
            difference_absorbance(test_sample_intensities, test_reference_intensities)