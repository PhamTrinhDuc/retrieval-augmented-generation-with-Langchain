import asyncio

def async_retry(max_retries: int = 3, # Số lần thử lại tối đa nếu hàm gặp lỗi. Mặc định là 3
                 delay: int = 1): # Thời gian chờ (tính bằng giây) giữa các lần thử lại. Mặc định là 1 giây.
    
    """decorator async_retry để tự động thử lại một hàm bất đồng bộ (asynchronous function) nếu nó gặp lỗi."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                
                except Exception as e:
                    print(f"Attempt {attempt} failed: {str(e)}")
                    await asyncio.sleep(delay)
            
            raise ValueError(f"Failed after {max_retries} attempts")
        
        return wrapper

    return decorator