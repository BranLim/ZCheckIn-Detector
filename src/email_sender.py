# email_sender.py
from threading import Thread
from email.mime.multipart import MIMEMultipart


class EmailSender:

    def __init__(self, email_from, to, cc=None, body=""):
        self.sender_email = email_from
        self.receiver_email = to
        self.cc = cc
        self.body = body

    def add_attachment(self, attachment=None):

        if self.attachments is None:
            self.attachments = []

        self.attachments.append(attachment)

        return self

    def send_email(self):
        t = Thread(target=self.send, args=())
        t.daemon = True
        t.start()
        return self

    def send(self):
        email_message = MIMEMultipart()
        email_message["From"] = self.sender_email
        email_message["Subject"] = "Employee Temperature Check-in"

        pass
