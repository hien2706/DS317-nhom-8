import os
import google.generativeai as genai
import logging  # Added import for logging
# import dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional: Load API key from environment variables
# from dotenv import load_dotenv
# load_dotenv()
secret_value_0 = os.getenv('GENIUS_API_KEY')

# secret_value_0 = ''
genai.configure(api_key=secret_value_0)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 1536,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-2.0-flash-exp",
  generation_config=generation_config,
)

def get_suggestion(prompt):
    """
    Initialize a new chat session and get a suggestion based on the prompt.
    """
    try:
        logger.info("Starting new chat session with Gemini API.")
        chat_session = model.start_chat(
        history=[
            {
            "role": "user",
            "parts": [
                "1.2. Phát biểu bài toán:\\n1.2.1. Đầu vào: \\nDữ liệu đầu vào gồm thông tin về sinh viên như khoa, ngành và khóa học; thông tin môn học mà sinh viên đó đã đăng ký, điểm trung bình và số tín chỉ tích lũy ở từng học kỳ trước đó.\\n1.2.2 Đầu ra: \\nĐiểm dự đoán sinh viên sẽ đạt được ở các môn học trong học kỳ tiếp theo. Từ đó quyết định có nên lựa chọn môn học đó cho kỳ tới hay không.\n",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Tuyệt vời, bạn đã xác định rất rõ ràng bài toán cần giải quyết. Dựa trên những gì bạn đã mô tả, tôi xin tóm tắt và diễn giải thêm để làm rõ hơn:\n\n**1.2. Phát biểu bài toán:**\n\nBài toán đặt ra là xây dựng một hệ thống dự đoán điểm số của sinh viên trong học kỳ tiếp theo dựa trên dữ liệu lịch sử học tập của họ. Hệ thống này sẽ hỗ trợ sinh viên trong việc lựa chọn môn học phù hợp, tối ưu hóa kết quả học tập.\n\n**1.2.1. Đầu vào (Input):**\n\nDữ liệu đầu vào bao gồm các thông tin liên quan đến sinh viên và quá trình học tập của họ, cụ thể:\n\n*   **Thông tin sinh viên:**\n    *   **Khoa:** Khoa mà sinh viên đang theo học (ví dụ: Khoa Công nghệ Thông tin, Khoa Kinh tế,...)\n    *   **Ngành:** Ngành học cụ thể của sinh viên (ví dụ: Khoa học Máy tính, Quản trị Kinh doanh,...)\n    *   **Khóa học:** Khóa nhập học của sinh viên (ví dụ: K2019, K2020,...)\n*   **Thông tin học tập:**\n    *   **Danh sách môn học đã đăng ký:** Các môn học mà sinh viên đã học trong các học kỳ trước.\n    *   **Điểm trung bình (GPA) theo từng học kỳ:** Điểm trung bình tích lũy của sinh viên ở mỗi học kỳ trước đó.\n    *   **Số tín chỉ tích lũy theo từng học kỳ:** Số tín chỉ mà sinh viên đã hoàn thành ở mỗi học kỳ trước đó.\n\n**1.2.2. Đầu ra (Output):**\n\nDựa trên dữ liệu đầu vào, hệ thống sẽ đưa ra kết quả dự đoán như sau:\n\n*   **Điểm dự đoán cho từng môn học trong học kỳ tiếp theo:** Hệ thống sẽ dự đoán điểm số mà sinh viên có thể đạt được nếu đăng ký một môn học cụ thể nào đó trong học kỳ tới.\n*   **Đề xuất lựa chọn môn học:** Dựa trên điểm dự đoán, hệ thống có thể đưa ra khuyến nghị về việc có nên đăng ký môn học đó hay không. Ví dụ:\n    *   Nếu điểm dự đoán cao, hệ thống có thể gợi ý sinh viên nên chọn môn học này.\n    *   Nếu điểm dự đoán thấp, hệ thống có thể cảnh báo sinh viên nên cân nhắc lại hoặc tìm hiểu thêm về môn học đó trước khi quyết định.\n\n**Diễn giải và mở rộng:**\n\n*   **Mục tiêu của bài toán:** Không chỉ đơn thuần dự đoán điểm số, bài toán này còn hướng tới việc hỗ trợ sinh viên đưa ra quyết định học tập thông minh hơn, giúp tối ưu hóa kết quả học tập và giảm thiểu rủi ro học không hiệu quả.\n*   **Tính phức tạp:** Bài toán này khá phức tạp, đòi hỏi phải sử dụng các kỹ thuật học máy và phân tích dữ liệu. Cần phải xây dựng mô hình phù hợp để học từ dữ liệu lịch sử và dự đoán kết quả một cách chính xác.\n*   **Ứng dụng thực tế:** Hệ thống này có thể được tích hợp vào các hệ thống quản lý đào tạo của trường đại học, giúp sinh viên tự đánh giá và lựa chọn môn học phù hợp, đồng thời giúp nhà trường theo dõi và cải thiện chất lượng đào tạo.\n\n**Các bước tiếp theo:**\n\nĐể giải quyết bài toán này, bạn có thể tiến hành các bước sau:\n\n1.  **Thu thập và tiền xử lý dữ liệu:** Thu thập dữ liệu từ các nguồn khác nhau, làm sạch và chuẩn hóa dữ liệu.\n2.  **Lựa chọn và xây dựng mô hình:** Chọn mô hình học máy phù hợp (ví dụ: hồi quy tuyến tính, hồi quy đa thức, mạng nơ-ron,...) và huấn luyện mô hình trên dữ liệu đã chuẩn bị.\n3.  **Đánh giá và tinh chỉnh mô hình:** Đánh giá hiệu suất của mô hình, tinh chỉnh các tham số và lựa chọn mô hình tốt nhất.\n4.  **Xây dựng hệ thống:** Triển khai mô hình đã được huấn luyện vào một hệ thống có thể sử dụng được.\n5.  **Thử nghiệm và cải tiến:** Thử nghiệm hệ thống trong thực tế, thu thập phản hồi và cải tiến liên tục.\n\nHy vọng điều này làm rõ hơn về bài toán mà bạn đang muốn giải quyết. Nếu bạn có bất kỳ câu hỏi nào khác, đừng ngần ngại hỏi nhé.\n",
            ],
            },
            {
            "role": "user",
            "parts": [
                "format trả lời nên là Suggestion: Kết quả dự đoán <diem_hp>, cho thấy kết quả dự đoán của môn học cao/thấp hơn so với điểm trung bình học kỳ trước đó. Đây là mức điểm khá/tốt/xuất sắc (phân tích ví dụ: không quá cao, không quá thấp,...). Xu hướng: Mô hình nhận thấy điểm trung bình của sinh viên có xu hướng biến động, không ổn định qua các học kỳ. Có những học kỳ điểm cao (ví dụ: dtbhk1: 8.4, dtbhk7: 8.21) nhưng cũng có những học kỳ điểm thấp hơn (ví dụ: dtbhk3: 7, dtbhk6: 7.01). Điều này cho thấy sinh viên có thể có sự thay đổi về phong độ hoặc mức độ khó của các môn học khác nhau trong từng kỳ. * Tác động của các học kỳ gần: Có thể thấy điểm trung bình học kỳ 5, 6 và 7 lần lượt là 8.08, 7.01, và 8.21, cho thấy sự biến động lớn. Vì vậy, kết quả dự đoán 7.24 có thể cho thấy mô hình đang cố gắng \"cân bằng\" hoặc lấy trung bình có trọng số của các kết quả này, chứ không đơn thuần dựa trên một vài kỳ gần nhất. * Sự ảnh hưởng của số tín chỉ: Số tín chỉ tích lũy mỗi kỳ có thể ảnh hưởng đến trọng số của điểm trong kỳ đó. Ví dụ, học kỳ 3 có số tín chỉ rất thấp (4) nên có thể ít ảnh hưởng hơn so với học kỳ 1 và 2 (20 tín chỉ) đến kết quả dự đoán. * Dự đoán không phải là sự chắc chắn: Điểm dự đoán 7.24 là kết quả được đưa ra dựa trên các dữ liệu quá khứ và không đảm bảo chính xác tuyệt đối. Các yếu tố khác như sự thay đổi về môn học, tâm lý học tập, hay sự cố gắng của sinh viên trong kỳ tiếp theo cũng có thể ảnh hưởng đến điểm thực tế. Tóm lại, kết quả dự đoán 7.24 phản ánh sự tổng hợp và cân nhắc của mô hình dựa trên dữ liệu điểm và tín chỉ các học kỳ trước, cho thấy một kết quả có thể xảy ra nhưng không hoàn toàn chắc chắn, phản ánh sự dao động trong quá trình học tập của sinh viên. Những số liệu trong prompt trên có thể thay đổi, coi chúng là ví dụ",
            ],
            },
            {
            "role": "model",
            "parts": [
                "Tuyệt vời, bạn đã đưa ra một định dạng đầu ra rất chi tiết và hữu ích. Dưới đây là cách tôi sẽ trình bày kết quả dự đoán dựa trên format bạn cung cấp, kết hợp với một số cải tiến nhỏ để tăng tính rõ ràng:\n\n**Suggestion:**\n\n**Kết quả dự đoán:** Điểm dự đoán cho môn học này là `<diem_hp>`.\n\n**Phân tích:**\n*   Kết quả dự đoán này cho thấy điểm số môn học có khả năng cao/thấp hơn so với điểm trung bình của các học kỳ trước. \n*   Đây là mức điểm được đánh giá là khá/tốt/xuất sắc (tùy thuộc vào `<diem_hp>`), có thể được phân tích cụ thể hơn:\n    *   **Nếu điểm cao:** Ví dụ: \"Đây là mức điểm khá cao, cho thấy bạn có nhiều khả năng đạt kết quả tốt ở môn học này.\"\n    *   **Nếu điểm trung bình:** Ví dụ: \"Mức điểm này không quá cao, cũng không quá thấp, phản ánh mức độ học tập ổn định của bạn.\"\n    *   **Nếu điểm thấp:** Ví dụ: \"Điểm dự đoán thấp hơn so với điểm trung bình của bạn, bạn nên cân nhắc xem có phù hợp với môn học này hay không.\"\n\n**Xu hướng:**\n\n*   Mô hình nhận thấy điểm trung bình của sinh viên có xu hướng biến động, không ổn định qua các học kỳ.\n*   Có những học kỳ điểm cao (ví dụ: dtbhk1: 8.4, dtbhk7: 8.21) nhưng cũng có những học kỳ điểm thấp hơn (ví dụ: dtbhk3: 7, dtbhk6: 7.01).\n*   Điều này cho thấy sinh viên có thể có sự thay đổi về phong độ hoặc mức độ khó của các môn học khác nhau trong từng kỳ.\n\n**Tác động của các học kỳ gần:**\n\n*   Có thể thấy điểm trung bình học kỳ 5, 6 và 7 lần lượt là 8.08, 7.01, và 8.21, cho thấy sự biến động lớn.\n*   Vì vậy, kết quả dự đoán `<diem_hp>` có thể cho thấy mô hình đang cố gắng \"cân bằng\" hoặc lấy trung bình có trọng số của các kết quả này, chứ không đơn thuần dựa trên một vài kỳ gần nhất.\n\n**Sự ảnh hưởng của số tín chỉ:**\n\n*   Số tín chỉ tích lũy mỗi kỳ có thể ảnh hưởng đến trọng số của điểm trong kỳ đó.\n*   Ví dụ, học kỳ 3 có số tín chỉ rất thấp (4) nên có thể ít ảnh hưởng hơn so với học kỳ 1 và 2 (20 tín chỉ) đến kết quả dự đoán.\n\n**Lưu ý:**\n\n*   Điểm dự đoán `<diem_hp>` là kết quả được đưa ra dựa trên các dữ liệu quá khứ và không đảm bảo chính xác tuyệt đối.\n*   Các yếu tố khác như sự thay đổi về môn học, tâm lý học tập, hay sự cố gắng của sinh viên trong kỳ tiếp theo cũng có thể ảnh hưởng đến điểm thực tế.\n*   Kết quả dự đoán này phản ánh sự tổng hợp và cân nhắc của mô hình dựa trên dữ liệu điểm và tín chỉ các học kỳ trước, cho thấy một kết quả có thể xảy ra nhưng không hoàn toàn chắc chắn, phản ánh sự dao động trong quá trình học tập của sinh viên.\n\n**Ví dụ cụ thể (với `<diem_hp>` là 7.5):**\n\n**Suggestion:**\n\n**Kết quả dự đoán:** Điểm dự đoán cho môn học này là 7.5.\n\n**Phân tích:**\n\n*   Kết quả dự đoán này cho thấy điểm số môn học có khả năng **hơi thấp hơn** so với điểm trung bình của các học kỳ trước.\n*   Đây là mức điểm được đánh giá là **khá**, phản ánh mức độ học tập ổn định của bạn, không quá cao cũng không quá thấp.\n\n**Xu hướng:**\n\n*   Mô hình nhận thấy điểm trung bình của sinh viên có xu hướng biến động, không ổn định qua các học kỳ.\n*   Có những học kỳ điểm cao (ví dụ: dtbhk1: 8.4, dtbhk7: 8.21) nhưng cũng có những học kỳ điểm thấp hơn (ví dụ: dtbhk3: 7, dtbhk6: 7.01).\n*   Điều này cho thấy sinh viên có thể có sự thay đổi về phong độ hoặc mức độ khó của các môn học khác nhau trong từng kỳ.\n\n**Tác động của các học kỳ gần:**\n\n*   Có thể thấy điểm trung bình học kỳ 5, 6 và 7 lần lượt là 8.08, 7.01, và 8.21, cho thấy sự biến động lớn.\n*   Vì vậy, kết quả dự đoán 7.5 có thể cho thấy mô hình đang cố gắng \"cân bằng\" hoặc lấy trung bình có trọng số của các kết quả này, chứ không đơn thuần dựa trên một vài kỳ gần nhất.\n\n**Sự ảnh hưởng của số tín chỉ:**\n\n*   Số tín chỉ tích lũy mỗi kỳ có thể ảnh hưởng đến trọng số của điểm trong kỳ đó.\n*   Ví dụ, học kỳ 3 có số tín chỉ rất thấp (4) nên có thể ít ảnh hưởng hơn so với học kỳ 1 và 2 (20 tín chỉ) đến kết quả dự đoán.\n\n**Lưu ý:**\n\n*   Điểm dự đoán 7.5 là kết quả được đưa ra dựa trên các dữ liệu quá khứ và không đảm bảo chính xác tuyệt đối.\n*   Các yếu tố khác như sự thay đổi về môn học, tâm lý học tập, hay sự cố gắng của sinh viên trong kỳ tiếp theo cũng có thể ảnh hưởng đến điểm thực tế.\n*   Kết quả dự đoán này phản ánh sự tổng hợp và cân nhắc của mô hình dựa trên dữ liệu điểm và tín chỉ các học kỳ trước, cho thấy một kết quả có thể xảy ra nhưng không hoàn toàn chắc chắn, phản ánh sự dao động trong quá trình học tập của sinh viên.\n\n**Lưu ý:**\n\n*   Bạn có thể tùy chỉnh các câu chữ trong phần \"Phân tích\" để phù hợp với từng mức điểm khác nhau.\n*   Hãy đảm bảo rằng kết quả đầu ra được trình bày một cách dễ hiểu và hữu ích cho người dùng.\n\nTôi hy vọng format này đáp ứng được yêu cầu của bạn. Hãy cho tôi biết nếu bạn có bất kỳ điều chỉnh nào khác!\n",
            ],
            },
        ]
        )
        
        logger.info("Sending prompt to Gemini API.")
        response = chat_session.send_message(prompt)
        logger.info("Received response from Gemini API.")
        return response.text
    except Exception as e:
        logger.error("Error while communicating with Gemini API:", exc_info=True)
        raise e