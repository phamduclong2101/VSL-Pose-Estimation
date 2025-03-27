
class ExpressionHandler:

    # MAPPING = {
    #     "Binh_thuong": "Ngồi yên 🤐",
    #     "cam_on": "Cảm ơn 😘",
    #     "Xin_chao": "Xin chào 🙋‍",
    #     "Tam_biet": "Tạm biệt 🤚",
    #     "yeu": "Yêu ❤️",
    #     "không": "Không 🤚",
    #     "Ban": "Bạn",
    #     "Ten": "Tên",
    #     "La_gi": "Là gì ?"

    # }
    MAPPING = {
        0: "Ban", 
        1: '',
        2: "Cam on", 
        3: 'F',
        4: "La gi ?", 
        5: "P", 
        6: 'T',
        7: "Tam biet", 
        8: "Ten", 
        9: "Toi", 
        10: 'Xin chao', 
        11: 'Yeu'}



    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        return ExpressionHandler.MAPPING[self.current_message]
