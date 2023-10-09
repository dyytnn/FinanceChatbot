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
from rasa_sdk.events import SlotSet, SessionStarted, ActionExecuted, EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, ConversationPaused

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


class ActionAskStartConversation(Action):
    def name(self) -> Text:
        return "action_start_conversation"

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
        if intent["name"] == "confirm_customer_false":
            # Khong phai khach hang
            dispatcher.utter_message(
                text=f"Dạ mình có quen ai tên {ne_name}  không ạ?")
        else:
            if DBD > 2:
                if TYPE_DBD == "NORMAL":
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} có hợp đồng Thẻ tín dụng sắp tới hạn thanh toán với tổng dư nợ trên sao kê là {ne_amount} đồng, {ne_title} có thể thu xếp đóng số tiền tối thiểu {ne_emi} đồng, trước ngày {ne_short_due_date} giúp bên em nha {ne_title}."
                    )

                    return []
                else:
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} có khoản vay trước đó và đã được chuyển đổi thành khoản trả góp trên Thẻ tín dụng, sắp tới hạn thanh toán với số tiền là {ne_amount} đồng. {ne_title} thu xếp đóng trước ngày {ne_short_due_date} giúp bên em nha {ne_title}."
                    )
                    return []

            if DBD == -1:
                if TYPE_DBD == "NORMAL":
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} có hợp đồng Thẻ tín dụng đã trễ hạn 1 ngày với tổng dư nợ trên sao kê là {ne_amount} đồng. Mình thu xếp đóng số tiền tối thiểu {ne_emi} trước 17g hôm nay cho bên em nha?"
                    )
                    return []
                else:
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} có khoản vay trước đó và được chuyển đổi thành khoản trả góp trên Thẻ tín dụng, đã trễ hạn 1 ngày. Mình thu xếp đóng số tiền {ne_title} trước 17g hôm nay cho bên em nha?"
                    )
                    return []

            if DBD == 0 or DBD == 1 or DBD == 2:
                if TYPE_DBD == "NORMAL":
                    dispatcher.utter_message(
                        text=f"Dạ,  {ne_title} có hợp đồng Thẻ tín dụng đã đến hạn thanh toán với tổng dư nợ trên sao kê là {ne_amount} đồng. Mình vui lòng thu xếp đóng số tiền tối thiểu {ne_emi} đồng trong ngày hôm nay giúp bên em nha {ne_title}."
                    )
                    return []
                else:
                    dispatcher.utter_message(
                        text=f"Dạ, {ne_title} có khoản vay trước đó và được chuyển đổi thành khoản trả góp trên Thẻ tín dụng, đã đến hạn thanh toán. Mình vui lòng thu xếp đóng số tiền {ne_amount} đồng trong ngày hôm nay giúp bên em nha {ne_title}."
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

        if "die" == intent["name"]:
            dispatcher.utter_message(
                text=f"Dạ em ghi nhận thông tin, và rất tiếc về điều này. Anh chị vui lòng liên hệ số tổng đài 1 9 0 0 6 9 3 9 cung cấp thông tin giấy báo tử để bên em có hướng giải quyết cho hồ sơ này. Cảm ơn và chào anh chị"
            )
            return []
        if "confirm_customer_connect_other_customer_false" == intent["name"]:
            # Khong quen khach hang
            dispatcher.utter_message(
                text="Dạ cho em xin phép hỏi, hiện tại mình dùng số điện thoại này bao lâu rồi?. Đã đăng ký sim chính chủ chưa?. Anh chị vui lòng cho em xin thông tin, vì số điện thoại này có trong hợp đồng Thẻ tín dụng bên công ty em ạ."
            )
        else:
            # Quen khach hang
            dispatcher.utter_message(
                text="Dạ, em là Thanh gọi đến từ công ty tài chính Finance, nhờ mình nhắn lại với [ne_title] [ne_name] giữ máy chờ liên hệ nhé. Em cảm ơn và xin chào ạ."
            )

        # check = False
        # Dạ mình có quen ai tên [ne_name] không ạ?

        # dispatcher.utter_message(
        #     template="utter_greet",

        # )


# Action 14
class ActionAskUserAboutThisNumberCustome14(Action):
    def name(self) -> Text:
        return "action_custom_14"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(
            text="Dạ bên em ghi nhận và sẽ kiểm tra lại thông tin của mình, nếu có thắc mắc anh chị liên hệ số 1 9 0 0  6 9 3 9 để bên em hỗ trợ thêm. Em cảm ơn và chào anh chị "
        )
        return [ConversationPaused()]


#  Repeat


class ActionWhenUserCannotHearClearly(Action):
    # can_not_hear_clearly
    def name(self) -> Text:
        return "action_can_not_hear_clearly"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        previousIntent = tracker.previous_message["intent"].get("name")
        print(previousIntent)
        # Nếu lập lại ý định ( cố tình / vô ý ) không hiểu vấn đề mà voicebot đang đưa ra thì chuyển qua action 3 ;
        if previousIntent == "can_not_hear_clearly":
            print("sdfsd")
        else:
            # Ngược lai nếu dưới 2 lần của dự định không hiểu ý đinh của voicebot thì sẽ nhắc lại
            # hành động (action) gần nhất  ( mới nhất )
            bot_event = next(e for e in reversed(
                tracker.events) if e["event"] == "bot")
            previousMessageFromBot = bot_event.get("name")
            dispatcher.utter_message(
                text=f"Dạ vâng, em xin nhắc lại thông tin đến {ne_title} "
                + f" {previousMessageFromBot}"
            )
        return []
