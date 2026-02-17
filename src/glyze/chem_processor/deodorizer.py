from glyze.glyceride_mix import GlycerideMix

class Deodorizer:

    @staticmethod
    def deoderization(mix: GlycerideMix, T: float, P: float) -> GlycerideMix:
        """
        Perform deoderization on a glyceride mix at a given temperature and pressure.

        Parameters:
        -----------
            mix (GlycerideMix): The glyceride mix to be deoderized.
            T (float): Temperature in Kelvin.
            P (float): Pressure in atm.

        Returns:
        -----------
            GlycerideMix: The deoderized glyceride mix.
        """
