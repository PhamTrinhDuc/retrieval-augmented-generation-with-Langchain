from pydantic import BaseModel

"""class xác định dạng query ở dạng text"""
class HospitalQueryInput(BaseModel):
    text: str
    
"""phản hồi được gửi lại cho người dùng của bạn bao gồm các trường input, output và intermediate_steps."""
class HospitalQueryOutput(BaseModel):
    input: str
    output: str
    intermediate: list[str]
