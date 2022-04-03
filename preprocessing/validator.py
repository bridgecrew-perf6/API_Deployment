from enum import Enum
from pydantic import BaseModel, Field, PositiveInt, ValidationError, validator
from typing import Literal, Optional, Union

def input_validator(json_):
    """
    This function check user input and raises error if the input is not expected format
    """

    class Feature(str, Enum):
        YES = "Yes"
        NO = "No"

    class Type(str, Enum):
        Apartment = "APARTMENT"
        House = "HOUSE"

    class Model(BaseModel):

        postcode: int = Field(
            ...,
            gt=999,
            lt=10000,
            description="The zip_code must be greater than 999 and less than 10,000",
        )
        kitchen_type: Literal["Not installed", "Semi equipped", "Equipped"] = Field(...)
        bedroom: Union[float, int] = Field(
            ...,
            ge=0,
            lt=25,
            description="The number of bedrooms must be greater or equal to zero",
        )
        building_condition: Literal[
            "As new",
            "Just renovated",
            "Good",
            "To be done up",
            "To renovate",
            "To restore",
        ] = Field(...)
        furnished: Feature = Optional[bool]
        terrace: Feature = Optional[bool]
        garden: Feature = Optional[bool]
        swimming_pool: Feature = Optional[bool]
        living_area: Union[float, int] = Field(
            ..., gt=0, description="The size in sqm must be greater than zero"
        )
        surface_plot: Union[float, int] = Field(None, gt=0, description="The size in sqm must be greater than zero"
        )
        property_type: Type = Field(...)


    try:
        Model(**json_)
        return "Excellent!"
    except ValidationError as e:
        capture_error = {}
        for json_ in e.errors():
            capture_error[json_["loc"][0]] = json_["msg"]

        return capture_error
