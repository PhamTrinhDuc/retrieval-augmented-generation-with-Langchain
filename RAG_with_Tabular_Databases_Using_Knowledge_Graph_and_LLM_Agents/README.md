### Project gồm 5 phần:

1. Tìm hiểu về các thành phần trong hệ thống RAG sử dạng Langchain bao gồm:
   - Chi tiết được lưu ở folder: 1_Get_familiar_with_langchain
   - Load data (CSV, PDF, ...)
   - Tạo prompt
   - Retrieval query sử dụng FAISS, ChromaDB, BM25, ....
   - Sử dụng LLM gọi API từ Groq
   - Tạo các agent cho từng từng nhiệm vụ cụ thể
   - Combine các thành phần trên để tạo thành 1 chatbot hoàn chỉnh.
     
2. Tìm hiểu về dữ liệu cho hệ thống.
   - Chi tiết được lưu ở folder: 2_Understand_data_&_buisiness_requirements
   - Data là các bảng dữ liệu tại bệnh viện gồm các thông tin về: bệnh nhân, bác sĩ, bệnh viện, đánh giá của bệnh nhân, ngày giờ cụ thể, ...
   - Tìm hiểu cách xây dựng hệ thống RAG và chức năng của các thành phần.

3. Tìm hiểu về Knowledge Graph sử dụng NEO4J.
   - Chi tiết đươc lưu trong folder: 3_Setup_Neo4j
   - Tìm hiểu các khái niệm cơ bản về graph database và hiết kế 1 Graph từ data ở bước 2. (đọc trong folder knowledge_base)
   - Build Graph từ data ở bước 2 và tìm hiểu các câu lệnh truy vấn tại <a href="https://console.neo4j.io/?product=aura-db&tenant=5612610d-24bf-47f7-8a5a-2e7c21a411d4">đây</a> (đọc code trong folder hospital_neo4j_etl)

4. Build chain và tạo agent cho hệ thống.
   - Chi tiết đươc lưu trong folder: 4_Build_Graph_RAG (chi tiết trong chatbot_api/src)
   - Build 1 chain_qa tương tự ở bước 1 (thay phần retrieval bằng retrieval của Neo4J), chain này có nhiệm vụ trả lời các câu hỏi review
   - Build 1 chain_cypher bằng cách cho 1 LLM tạo câu truy vấn và truy vấn trong graph từ kết quả đó 1 LLM khác sẽ dựa vào đó và đưa ra response cuối.
   - Tạo 1 agent có 4 chức năng: 2 tool từ 2 chain vừa được tạo, 1 tool trả về thời gian chờ của bệnh viện và 1 tool trả về bệnh viện có thời gian chờ ngắn nhất.
  
5. Phần cuối cùng Build lại 1 hệ thống hoàn chỉnh end-to-end.
   - Chi tiết trong folder 5_Deploy_Langchain_Agent
   - folder gồm 3 phần: từ phần 3 + phần 4 và thêm 1 phần deloy lên FastAPI và Streamlist.
