# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

# from typing import Any, Text, Dict, List
#
# from rasa_sdk import Action, Tracker
# from rasa_sdk.executor import CollectingDispatcher
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []
import pandas as pd
from typing import Text, List, Dict, Any
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType,ConversationPaused
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ConversationPaused,UserUttered
df = pd.read_csv("customer1.csv", encoding="utf-8")
print(df.loc[1], df.loc[1].loc["STT"])

index = 1

customer = df.loc[1]

ID_USER = customer.loc["ID_USER"]
LastName = customer.loc["LastName"]
FirstName = customer.loc["FirstName"]
ne_name = FirstName + " " + LastName
ne_title = "anh" if customer.loc["Gender"] == "Male" else "chị"
ne_short_due_date = customer.loc["DeadlineDate"]
Phone = customer.loc["Phone"]

DBD = int(customer.loc["DBD"])
TYPE_DBD = customer.loc["TYPE_DBD"]
ne_amount = customer.loc["Total"]
ne_emi = customer.loc["Available"]


#  bắt dầu
# class ActionSessionStart(Action):
#     def name(self) -> Text:
#         return "action_session_start"

#     def run(
#         self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
#     ) -> List[Dict[Text, Any]]:
#         metadata = tracker.get_slot("session_started_metadata")

#         # Do something with the metadata
#         print(metadata)
#         dispatcher.utter_message(
#             text="Chào anh/chị {} . Em là Thanh gọi từ công ty tài chính Finance, rất vui được kết nối với {}. Cho em xin ít phút để trao đổi thông tin được không ạ?".format(
#                 "Gia Thuận", "Gia Thuận"
#             )
#         )
#         # the session should begin with a `session_started` event and an `action_listen`
#         # as a user message follows
#         return [SessionStarted(), ActionExecuted("action_listen")]

# class ActionInitStart(Action):
#     def name(self) -> Text:
#         """This is the name to be mentioned in domain.yml and stories.md files
#             for this action."""
#         return "action_session_start"

#     async def run(
#             self,
#             dispatcher: CollectingDispatcher,
#             tracker,
#             domain: Dict[Text, Any],
#     ) -> List[EventType]:
#         """This run function will be executed when "action_session_start" is
#             triggered."""
#         # The session should begin with a 'session_started' event
#         events = [SessionStarted()]
        
#         dispatcher.utter_message(
#             text="Em chào anh/chị ạ? Em là nhân viên của công ty .....Anh/Chị có thể cho em ít thời gian trao đổi được không ạ ?")
#         # events.append(ActionExecuted("action_listen"))
#         return events

class ActionAskStartConversation(Action):
    def name(self) -> Text:
        return "action_session_start"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # text = tracker.latest_message['text']
        # text_input = text.lower()
        
        intent = tracker.latest_message["intent"]
        # check = False
        print(intent)
        
       
        dispatcher.utter_message(
            text=f"Chào {ne_title} {ne_name} . Em là Thanh gọi từ công ty tài chính Finance, rất vui được kết nối với {ne_name}. Cho em xin ít phút để trao đổi thông tin được không ạ?"
        )
        # dispatcher.utter_message(
        #     template="utter_greet",

        # )
        return []


# Xác nhận có phải khách hàng hay không ?


class ActionAskIfUserIsCustomerOrNot(Action):
    def name(self) -> Text:
        return "action_confirm_is_customer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # text = tracker.latest_message['text']
        # text_input = text.lower()

        # check = False
        # print(text_input)
        intent = tracker.latest_message["intent"]
        print(intent)
        if intent["name"] == "xac_nhan_sai":
            # Khong phai khach hang
            dispatcher.utter_message(
                text=f"Dạ mình có quen ai tên {ne_name}  không ạ?")
            return []
        elif intent["name"]=="ban":
            dispatcher.utter_message(text="Dạ vâng để em gọi lại sau")
            return []
        else:
            if DBD > 2:
                if TYPE_DBD == "NORMAL":
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} ơi hợp đồng Thẻ tín dụng sắp tới hạn thanh toán với tổng dư nợ trên sao kê là {ne_amount} đồng, {ne_title} số tiền tối thiểu {ne_emi} đồng, hạn ngày {ne_short_due_date} không biết hợp đồng này có phải của anh chị không ạ "
                    )

                    return []
                else:
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} ơi khoản vay đã được chuyển đổi thành khoản trả góp trên Thẻ tín dụng, sắp tới hạn thanh toán với số tiền là {ne_amount} đồng. {ne_title} hạn ngày {ne_short_due_date} ,không biết khoản vay này có phải của anh chị không ạ"
                    )
                    return []

            if DBD == -1:
                if TYPE_DBD == "NORMAL":
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} ơi hợp đồng Thẻ tín dụng với tổng dư nợ trên sao kê là {ne_amount} đồng. số tiền tối thiểu {ne_emi} đồng,không biết hợp đồng này có phải của {ne_title} không ạ, vì hợp đồng bị trễ hạn rồi ạ"
                    )
                    return []
                else:
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} ơi  khoản vay được chuyển đổi thành khoản trả góp trên thẻ tín dụng khoản nợ là {ne_amount} đồng.không biết khoản vay này có phải của anh chị không ạ,vì khoản vay hiện tại đang bị trễ hạn rồi ạ "
                    )
                    return []

            if DBD == 0 or DBD == 1 or DBD == 2:
                if TYPE_DBD == "NORMAL":
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} ơi  hợp đồng Thẻ tín dụng đã đến hạn thanh toán với tổng dư nợ trên sao kê là {ne_amount} đồng,số tiền tối thiểu phải trả là {ne_emi} đồng không biết hợp đồng này có phải của anh chị không ạ"
                    )
                    return []
                else:
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} ơi khoản vay được chuyển đổi thành khoản trả góp trên Thẻ tín dụng, đã đến hạn thanh toán. Tổng dư nợ là {ne_amount} đồng trong,không biết hợp đồng này có phải của anh chị không ạ."
                    )
                    return []

        # check = False
        # Dạ mình có quen ai tên [ne_name] không ạ?
        # dispatcher.utter_message(text="Xac nhan khach hang")

        # dispatcher.utter_message(
        #     template="utter_greet",

        # )
        return []


# Xác nhận người đang gọi có quen khách hàng có trong danh sách hay không ?


class ActionAskIfThisUserConnectWithThisCustomer(Action):
    def name(self) -> Text:
        return "action_confirm_customer_connect_other_customer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # text = tracker.latest_message['text']
        # text_input = text.lower()

        # check = False
        # print(text_input)
        intent = tracker.latest_message["intent"]

        print(intent)

        if  intent["name"]=="xac_nhan_sai":
            dispatcher.utter_message(
                text=f"Dạ cho em xin phép hỏi, hiện tại mình dùng số điện thoại này bao lâu rồi?. Đã đăng ký sim chính chủ chưa?. Anh chị vui lòng cho em xin thông tin, vì số điện thoại này có trong hợp đồng Thẻ tín dụng bên công ty em ạ."
            )
            return []
        if  intent["name"]=="xac_nhan_dung":
            # Khong quen khach hang
            entities=tracker.latest_message["entities"]
            # if entities["entity"]=="confirmation":
            dispatcher.utter_message(
                    text=f"Dạ vậy nhờ anh chị nhắn lại với {ne_name} vui lòng thu xếp đóng số tiền {ne_amount} đồng trong hạn giúp bên em nha "
                )
            return []
        # else:
        #     # Quen khach hang
        #     dispatcher.utter_message(
        #         text="Dạ, em là Thanh gọi đến từ công ty tài chính Finance, nhờ mình nhắn lại với [ne_title] [ne_name] giữ máy chờ liên hệ nhé. Em cảm ơn và xin chào ạ."
        #     )
        return []
        # check = False
        # Dạ mình có quen ai tên [ne_name] không ạ?

        # dispatcher.utter_message(
        #     template="utter_greet",

        # )

# Xác nhận người đang gọi có phải hợp đồng này không ?


class ActionAskIfThisUserConnectWithThisontract(Action):
    def name(self) -> Text:
        return "action_confirm_customer_connect_contract"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # text = tracker.latest_message['text']
        # text_input = text.lower()

        # check = False
        # print(text_input)
        intent = tracker.latest_message["intent"]

        print(intent)

        if  intent["name"]=="xac_nhan_sai":
            dispatcher.utter_message(
                text=f"Dạ mình có quen ai tên {ne_name} không ạ?")
            return [ActionExecuted("action_listen"),ActionAskIfThisUserConnectWithThisCustomer]
        if  intent["name"]=="xac_nhan_dung":
            # Khong quen khach hang
            entities=tracker.latest_message["entities"]
            # if entities["entity"]=="confirmation":
            dispatcher.utter_message(
                    text=f"Dạ vậy nhờ {ne_title}  vui lòng thu xếp đóng số tiền {ne_amount} đồng trong hạn giúp bên em nha "
                )
            return []
        if intent["name"]=="thac_mac":
            
            dispatcher.utter_message(text="Bạn có thể xem lại thông tin chi tiết hợp đồng tại đây",json_message={
                "ID_USER": f"{ID_USER}",
                "Họ tên": f"{ne_name}",
                "Điện thoại": f"{Phone}",
                "DBD": f"{DBD}",
                "Type DBD": f"{TYPE_DBD}",
                "Tổng tiền": f"{ne_amount}",
                "Ngày Hạn": f"{ne_short_due_date}",
            })
            return []
            # ID_USER = customer.loc["ID_USER"]
            # LastName = customer.loc["LastName"]
            # FirstName = customer.loc["FirstName"]
            # ne_name = FirstName + " " + LastName
            # ne_title = "anh" if customer.loc["Gender"] == "Male" else "chị"
            # ne_short_due_date = customer.loc["DeadlineDate"]
            # Phone = customer.loc["Phone"]

            # DBD = int(customer.loc["DBD"])
            # TYPE_DBD = customer.loc["TYPE_DBD"]
            # ne_amount = customer.loc["Total"]
            # ne_emi = customer.loc["Available"]
        # else:
        #     # Quen khach hang
        #     dispatcher.utter_message(
        #         text="Dạ, em là Thanh gọi đến từ công ty tài chính Finance, nhờ mình nhắn lại với [ne_title] [ne_name] giữ máy chờ liên hệ nhé. Em cảm ơn và xin chào ạ."
        #     )
        return []
    

class ActionAskIfThisUserConfirmToPay(Action):
    def name(self) -> Text:
        return "action_confirm_to_pay"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # text = tracker.latest_message['text']
        # text_input = text.lower()

        # check = False
        # print(text_input)
        intent = tracker.latest_message["intent"]

        print(intent)

        if  intent["name"]=="tra":
            dispatcher.utter_message(
                text=f"Dạ vậy anh chị đóng trước {ne_short_due_date} giúp em em cảm ơn")    
            return []
        

        if  intent["name"]=="khong_tra":
            # Khong quen khach hang
            entities=tracker.latest_message["entities"]
            # if entities["entity"]=="confirmation":
            dispatcher.utter_message(
                    text=f"Dạ {ne_title} có chuyện gì không ạ,{ne_title} có thể chia sẻ với em để em hỗ trợ anh chị ,anh chị có thể trả cho em một khoản nhất định trước cũng được ạ"
                )
            return []

            
        return []
    

class ActionAskIfThisUserConvinceToPay(Action):
    def name(self) -> Text:
        return "action_convince_to_pay"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # text = tracker.latest_message['text']
        # text_input = text.lower()

        # check = False
        # print(text_input)
        intent = tracker.latest_message["intent"]

        print(intent)

        if  intent["name"]=="tra":
            dispatcher.utter_message(text="Bạn có thể xem lại thông tin chi tiết hợp đồng tại đây",json_message={
                "ID_USER": f"{ID_USER}",
                "Họ tên": f"{ne_name}",
                "Điện thoại": f"{Phone}",
                "DBD": f"{DBD}",
                "Type DBD": f"{TYPE_DBD}",
                "Tổng tiền": f"{ne_amount}",
                "Ngày Hạn": f"{ne_short_due_date}",
            })
            # dispatcher.utter_message(
            #     text=f"Dạ vậy anh chị đóng trước {ne_short_due_date} giúp em em cảm ơn")    
            return []
        

        if  intent["name"]=="khong_tra":
            # Khong quen khach hang
            entities=tracker.latest_message["entities"]
            # if entities["entity"]=="confirmation":
            dispatcher.utter_message(
                    text=f"Trong trường hợp này chúng tôi sẽ nhờ pháp luật can thiệp"
                )
            return []

            
        return []
# # Action 14
# class ActionAskUserAboutThisNumberCustome14(Action):
#     def name(self) -> Text:
#         return "action_custom_14"

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         dispatcher.utter_message(
#             text="Dạ bên em ghi nhận và sẽ kiểm tra lại thông tin của mình, nếu có thắc mắc anh chị liên hệ số 1 9 0 0  6 9 3 9 để bên em hỗ trợ thêm. Em cảm ơn và chào anh chị "
#         )
#         return [ConversationPaused()]


# #  Repeat


# class ActionWhenUserCannotHearClearly(Action):
#     # can_not_hear_clearly
#     def name(self) -> Text:
#         return "action_can_not_hear_clearly"

#     def run(
#         self,
#         dispatcher: CollectingDispatcher,
#         tracker: Tracker,
#         domain: Dict[Text, Any],
#     ) -> List[Dict[Text, Any]]:
#         previousIntent = tracker.previous_message["intent"].get("name")
#         print(previousIntent)
#         # Nếu lập lại ý định ( cố tình / vô ý ) không hiểu vấn đề mà voicebot đang đưa ra thì chuyển qua action 3 ;
#         if previousIntent == "can_not_hear_clearly":
#             print("sdfsd")
#         else:
#             # Ngược lai nếu dưới 2 lần của dự định không hiểu ý đinh của voicebot thì sẽ nhắc lại
#             # hành động (action) gần nhất  ( mới nhất )
#             bot_event = next(e for e in reversed(
#                 tracker.events) if e["event"] == "bot")
#             previousMessageFromBot = bot_event.get("name")
#             dispatcher.utter_message(
#                 text=f"Dạ vâng, em xin nhắc lại thông tin đến {ne_title} "
#                 + f" {previousMessageFromBot}"
#             )
#         return []
