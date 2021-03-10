import unittest
from agndiag import __version__
from agndiag.mpa_jhu import diag_nii_Sabater2012, diag_oi_Sabater2012, diag_sii_Sabater2012


def test_version():
    assert __version__ == "0.1.0"


class TestSabater2012(unittest.TestCase):
    """
    Test the classification methods of Sabater et al. 2012.
    Check the classification of galaxies depending on their possition on the diagram.
    Codes:
    0 - Unclassified
    1 - SFN
    2 - Seyfert
    3 - LINER
    4 - TO
    5 - NLAGN (Seyfert or LINER)
    8 - TO or SFN
    9 - TO or NLAGN
    10 - Seyfert 1 (not used here)
    """

    def setUp(self):
        pass

    def test_nii(self):
        """
        Test the position for detections in the [NII] diagram
        """
        test = [
            [[-0.1], [0.7], [0], [0], 5, "AGN norm"],
            [[0.1], [0.7], [0], [0], 5, "AGN right"],
            [[-0.1], [0.9], [0], [0], 5, "AGN up"],
            [[0.1], [0.9], [0], [0], 5, "AGN right up"],
            [[-0.1], [0.0], [0], [0], 4, "TO"],
            [[-0.5], [0.0], [0], [0], 1, "SFN"],
        ]
        for t in test:
            diag, c_diag = diag_nii_Sabater2012(t[0], t[1], t[2], t[3])
            self.assertEqual(diag[0], t[4])

    def test_nii_limits(self):
        """
        Test the position for detections in the [NII] diagram using limits
        """
        test = [
            [[-0.1], [0.7], [1], [2], 0, "AGN norm 1"],
            [[-0.1], [0.7], [0], [2], 5, "AGN norm 2"],
            [[-0.1], [0.7], [2], [2], 5, "AGN norm 3"],
            [[-0.1], [0.7], [2], [0], 5, "AGN norm 4"],
            [[-0.1], [0.7], [2], [1], 0, "AGN norm 5"],
            [[-0.1], [0.7], [0], [1], 0, "AGN norm 6"],
            [[-0.1], [0.7], [1], [1], 0, "AGN norm 7"],
            [[-0.1], [0.7], [1], [0], 0, "AGN norm 8"],
            [[0.1], [0.7], [1], [2], 0, "AGN right 1"],
            [[0.1], [0.7], [0], [2], 5, "AGN right 2"],
            [[0.1], [0.7], [2], [2], 5, "AGN right 3"],
            [[0.1], [0.7], [2], [0], 5, "AGN right 4"],
            [[0.1], [0.7], [2], [1], 5, "AGN right 5"],
            [[0.1], [0.7], [0], [1], 5, "AGN right 6"],
            [[0.1], [0.7], [1], [1], 0, "AGN right 7"],
            [[0.1], [0.7], [1], [0], 0, "AGN right 8"],
            [[-0.1], [0.9], [1], [2], 5, "AGN up 1"],
            [[-0.1], [0.9], [0], [2], 5, "AGN up 2"],
            [[-0.1], [0.9], [2], [2], 5, "AGN up 3"],
            [[-0.1], [0.9], [2], [0], 5, "AGN up 4"],
            [[-0.1], [0.9], [2], [1], 0, "AGN up 5"],
            [[-0.1], [0.9], [0], [1], 0, "AGN up 6"],
            [[-0.1], [0.9], [1], [1], 0, "AGN up 7"],
            [[-0.1], [0.9], [1], [0], 5, "AGN up 8"],
            [[0.1], [0.9], [1], [2], 5, "AGN right up 1"],
            [[0.1], [0.9], [0], [2], 5, "AGN right up 2"],
            [[0.1], [0.9], [2], [2], 5, "AGN right up 3"],
            [[0.1], [0.9], [2], [0], 5, "AGN right up 4"],
            [[0.1], [0.9], [2], [1], 5, "AGN right up 5"],
            [[0.1], [0.9], [0], [1], 5, "AGN right up 6"],
            [[0.1], [0.9], [1], [1], 0, "AGN right up 7"],
            [[0.1], [0.9], [1], [0], 5, "AGN right up 8"],
            [[-0.1], [0.0], [1], [2], 0, "TO 1"],
            [[-0.1], [0.0], [0], [2], 9, "TO 2"],
            [[-0.1], [0.0], [2], [2], 9, "TO 3"],
            [[-0.1], [0.0], [2], [0], 9, "TO 4"],
            [[-0.1], [0.0], [2], [1], 0, "TO 5"],
            [[-0.1], [0.0], [0], [1], 8, "TO 6"],
            [[-0.1], [0.0], [1], [1], 8, "TO 7"],
            [[-0.1], [0.0], [1], [0], 8, "TO 8"],
            [[-0.5], [0.0], [1], [2], 0, "SFN 1"],
            [[-0.5], [0.0], [0], [2], 0, "SFN 2"],
            [[-0.5], [0.0], [2], [2], 0, "SFN 3"],
            [[-0.5], [0.0], [2], [0], 0, "SFN 4"],
            [[-0.5], [0.0], [2], [1], 0, "SFN 5"],
            [[-0.5], [0.0], [0], [1], 1, "SFN 6"],
            [[-0.5], [0.0], [1], [1], 1, "SFN 7"],
            [[-0.5], [0.0], [1], [0], 1, "SFN 8"],
        ]
        for t in test:
            diag, c_diag = diag_nii_Sabater2012(t[0], t[1], t[2], t[3], use_limits=True)
            self.assertEqual(diag[0], t[4])

    def test_sii(self):
        """
        Test the position for detections in the [SII] diagram
        """
        test = [
            [[-0.0], [0.0], [0], [0], 3, "LINER norm"],
            [[0.6], [0.0], [0], [0], 3, "LINER right"],
            [[0.0], [1.5], [0], [0], 2, "Seyfert"],
            [[0.4], [3.5], [0], [0], 2, "Seyfert up"],  # Check position
            [[-0.5], [0.0], [0], [0], 1, "SFN"],
        ]
        for t in test:
            diag, c_diag = diag_sii_Sabater2012(t[0], t[1], t[2], t[3])
            self.assertEqual(diag[0], t[4])

    def test_sii_limits(self):
        """
        Test the position for detections in the [SII] diagram using limits
        """
        test = [
            [[-0.0], [0.0], [1], [2], 0, "LINER norm 1"],
            [[-0.0], [0.0], [0], [2], 5, "LINER norm 2"],
            [[-0.0], [0.0], [2], [2], 5, "LINER norm 3"],
            [[-0.0], [0.0], [2], [0], 3, "LINER norm 4"],
            [[-0.0], [0.0], [2], [1], 0, "LINER norm 5"],
            [[-0.0], [0.0], [0], [1], 0, "LINER norm 6"],
            [[-0.0], [0.0], [1], [1], 0, "LINER norm 7"],
            [[-0.0], [0.0], [1], [0], 0, "LINER norm 8"],
            [[0.6], [0.0], [1], [2], 0, "LINER right 1"],
            [[0.6], [0.0], [0], [2], 5, "LINER right 2"],
            [[0.6], [0.0], [2], [2], 5, "LINER right 3"],
            [[0.6], [0.0], [2], [0], 3, "LINER right 4"],
            [[0.6], [0.0], [2], [1], 3, "LINER right 5"],
            [[0.6], [0.0], [0], [1], 3, "LINER right 6"],
            [[0.6], [0.0], [1], [1], 0, "LINER right 7"],
            [[0.6], [0.0], [1], [0], 0, "LINER right 8"],
            [[0.0], [1.5], [1], [2], 0, "Seyfert 1"],
            [[0.0], [1.5], [0], [2], 2, "Seyfert 2"],
            [[0.0], [1.5], [2], [2], 5, "Seyfert 3"],
            [[0.0], [1.5], [2], [0], 5, "Seyfert 4"],
            [[0.0], [1.5], [2], [1], 0, "Seyfert 5"],
            [[0.0], [1.5], [0], [1], 0, "Seyfert 6"],
            [[0.0], [1.5], [1], [1], 0, "Seyfert 7"],
            [[0.0], [1.5], [1], [0], 0, "Seyfert 8"],
            [[0.4], [3.5], [1], [2], 0, "Seyfert up 1"],
            [[0.4], [3.5], [0], [2], 2, "Seyfert up 2"],
            [[0.4], [3.5], [2], [2], 5, "Seyfert up 3"],
            [[0.4], [3.5], [2], [0], 5, "Seyfert up 4"],
            [[0.4], [3.5], [2], [1], 5, "Seyfert up 5"],
            [[0.4], [3.5], [0], [1], 5, "Seyfert up 6"],
            [[0.4], [3.5], [1], [1], 0, "Seyfert up 7"],
            [[0.4], [3.5], [1], [0], 0, "Seyfert up 8"],
            [[-0.5], [0.0], [1], [2], 0, "SFN 1"],
            [[-0.5], [0.0], [0], [2], 0, "SFN 2"],
            [[-0.5], [0.0], [2], [2], 0, "SFN 3"],
            [[-0.5], [0.0], [2], [0], 0, "SFN 4"],
            [[-0.5], [0.0], [2], [1], 0, "SFN 5"],
            [[-0.5], [0.0], [0], [1], 1, "SFN 6"],
            [[-0.5], [0.0], [1], [1], 1, "SFN 7"],
            [[-0.5], [0.0], [1], [0], 1, "SFN 8"],
        ]
        for t in test:
            diag, c_diag = diag_sii_Sabater2012(t[0], t[1], t[2], t[3], use_limits=True)
            self.assertEqual(diag[0], t[4])

    def test_oi(self):
        """
        Test the position for detections in the [OI] diagram
        """
        test = [
            [[-0.7], [-0.2], [0], [0], 3, "LINER norm"],
            [[-0.5], [0.0], [0], [0], 3, "LINER right"],
            [[-1.0], [0.5], [0], [0], 2, "Seyfert"],
            [[-0.5], [3.5], [0], [0], 2, "Seyfert up"],  # Check position
            [[-1.5], [-0.5], [0], [0], 1, "SFN"],
        ]
        for t in test:
            diag, c_diag = diag_oi_Sabater2012(t[0], t[1], t[2], t[3])
            self.assertEqual(diag[0], t[4])

    def test_oi_limits(self):
        """
        Test the position for detections in the [OI] diagram using limits
        """
        test = [
            [[-0.7], [-0.2], [1], [2], 0, "LINER norm 1"],
            [[-0.7], [-0.2], [0], [2], 5, "LINER norm 2"],
            [[-0.7], [-0.2], [2], [2], 5, "LINER norm 3"],
            [[-0.7], [-0.2], [2], [0], 3, "LINER norm 4"],
            [[-0.7], [-0.2], [2], [1], 0, "LINER norm 5"],
            [[-0.7], [-0.2], [0], [1], 0, "LINER norm 6"],
            [[-0.7], [-0.2], [1], [1], 0, "LINER norm 7"],
            [[-0.7], [-0.2], [1], [0], 0, "LINER norm 8"],
            [[-0.5], [0.0], [1], [2], 0, "LINER right 1"],
            [[-0.5], [0.0], [0], [2], 5, "LINER right 2"],
            [[-0.5], [0.0], [2], [2], 5, "LINER right 3"],
            [[-0.5], [0.0], [2], [0], 3, "LINER right 4"],
            [[-0.5], [0.0], [2], [1], 3, "LINER right 5"],
            [[-0.5], [0.0], [0], [1], 3, "LINER right 6"],
            [[-0.5], [0.0], [1], [1], 0, "LINER right 7"],
            [[-0.5], [0.0], [1], [0], 0, "LINER right 8"],
            [[-1.0], [0.5], [1], [2], 0, "Seyfert 1"],
            [[-1.0], [0.5], [0], [2], 2, "Seyfert 2"],
            [[-1.0], [0.5], [2], [2], 5, "Seyfert 3"],
            [[-1.0], [0.5], [2], [0], 5, "Seyfert 4"],
            [[-1.0], [0.5], [2], [1], 0, "Seyfert 5"],
            [[-1.0], [0.5], [0], [1], 0, "Seyfert 6"],
            [[-1.0], [0.5], [1], [1], 0, "Seyfert 7"],
            [[-1.0], [0.5], [1], [0], 0, "Seyfert 8"],
            [[-0.5], [3.5], [1], [2], 0, "Seyfert up 1"],
            [[-0.5], [3.5], [0], [2], 2, "Seyfert up 2"],
            [[-0.5], [3.5], [2], [2], 5, "Seyfert up 3"],
            [[-0.5], [3.5], [2], [0], 5, "Seyfert up 4"],
            [[-0.5], [3.5], [2], [1], 5, "Seyfert up 5"],
            [[-0.5], [3.5], [0], [1], 5, "Seyfert up 6"],
            [[-0.5], [3.5], [1], [1], 0, "Seyfert up 7"],
            [[-0.5], [3.5], [1], [0], 0, "Seyfert up 8"],
            [[-1.5], [-0.5], [1], [2], 0, "SFN 1"],
            [[-1.5], [-0.5], [0], [2], 0, "SFN 2"],
            [[-1.5], [-0.5], [2], [2], 0, "SFN 3"],
            [[-1.5], [-0.5], [2], [0], 0, "SFN 4"],
            [[-1.5], [-0.5], [2], [1], 0, "SFN 5"],
            [[-1.5], [-0.5], [0], [1], 1, "SFN 6"],
            [[-1.5], [-0.5], [1], [1], 1, "SFN 7"],
            [[-1.5], [-0.5], [1], [0], 1, "SFN 8"],
        ]
        for t in test:
            diag, c_diag = diag_oi_Sabater2012(t[0], t[1], t[2], t[3], use_limits=True)
            self.assertEqual(diag[0], t[4])

    def test_nii_real_data_limits(self):
        """
        Test the position for real data in the [NII] diagram using limits
        """
        test = [
            [
                [-0.46892769992791428],
                [-0.82047634483325904],
                [0],
                [1],
                1,
                "SFN down arrow",
            ],
        ]
        for t in test:
            diag, c_diag = diag_nii_Sabater2012(t[0], t[1], t[2], t[3], use_limits=True)
            self.assertEqual(diag[0], t[4])


if __name__ == "__main__":
    unittest.main()
