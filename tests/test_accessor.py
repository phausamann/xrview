import pytest
import xarray as xr

from xrview.accessors import _dict_from_da


@pytest.fixture()
def airtemps():
    """"""
    return xr.tutorial.open_dataset("air_temperature")


@pytest.fixture()
def air(airtemps):
    """"""
    air = airtemps.air - 273.15
    # copy attributes to get nice figure labels and change Kelvin to Celsius
    air.attrs = airtemps.air.attrs
    air.attrs["units"] = "deg C"

    return air


class TestUtils:
    def test_dict_from_da(self, airtemps):
        """"""
        air = airtemps.air[:3, :3, :100]

        da_dict = _dict_from_da(air)
        assert set(da_dict.keys()) == {"time_lat_lon", "air"}

        da_dict = _dict_from_da(air, unstack="lat")
        assert set(da_dict.keys()) == {
            "time_lon",
            "air_70.0",
            "air_72.5",
            "air_75.0",
        }

        da_dict = _dict_from_da(air, unstack=["lat", "lon"])
        assert "time" in da_dict
        assert "air_70.0_200.0" in da_dict

        da_dict = _dict_from_da(air, unstack=["lon", "lat"])
        assert "time" in da_dict
        assert "air_200.0_70.0" in da_dict


class TestExamples:
    def test_line(self, air):
        """"""
        air1d = air.isel(lat=10, lon=10)
        air1d.view.line()

        air.isel(lon=10, lat=[19, 21, 22]).view.line(hue="lat")
