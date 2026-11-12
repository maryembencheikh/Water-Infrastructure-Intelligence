from typing import Optional,Literal
from pydantic import BaseModel, ConfigDict


class WaterPointInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int

    amount_tsh: Optional[float] = None
    date_recorded: Optional[str] = None
    funder: Optional[str] = None
    gps_height: Optional[int] = None
    installer: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    wpt_name: Optional[str] = None
    num_private: Optional[int] = None
    basin: Optional[str] = None
    subvillage: Optional[str] = None
    region: Optional[str] = None
    region_code: Optional[int] = None
    district_code: Optional[int] = None
    lga: Optional[str] = None
    ward: Optional[str] = None
    population: Optional[int] = None
    public_meeting: Optional[bool] = None
    recorded_by: Optional[str] = None
    scheme_management: Optional[str] = None
    scheme_name: Optional[str] = None
    permit: Optional[bool] = None
    construction_year: Optional[int] = None
    extraction_type: Optional[str] = None
    extraction_type_group: Optional[str] = None
    extraction_type_class: Optional[str] = None
    management: Optional[str] = None
    management_group: Optional[str] = None
    payment: Optional[str] = None
    payment_type: Optional[str] = None
    water_quality: Optional[str] = None
    quality_group: Optional[str] = None
    quantity: Optional[str] = None
    quantity_group: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    source_class: Optional[str] = None
    waterpoint_type: Optional[str] = None
    waterpoint_type_group: Optional[str] = None




class PredictionResponse(BaseModel):
    id: int
    prediction: Literal[
        "functional",
        "functional needs repair",
        "non functional"
    ]
    probabilities: dict[str, float]
    status: Literal["success"] = "success"
