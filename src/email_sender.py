# email_sender.py

import os
import logging
import constants
from smtplib import SMTP_SSL
from threading import Thread
from email.mime.multipart import MIMEMultipart
from email.utils import COMMASPACE
from logging.handlers import TimedRotatingFileHandler


current_app_path = os.path.abspath(os.path.dirname(__file__))

class EmailSender:

    logger = logging.getLogger("Email Sender")

    def __init__(self, email_from, email_to, cc=None, body=""):
        assert isinstance(email_to, list)
        
        self.setup_logging()

        self.sender_email = email_from
        self.receiver_email = email_to
        self.cc = cc
        self.body = body


    def setup_logging(self):

        log_folder = os.path.join(current_app_path, f'../{constants.LOG_FOLDER}')
        try:
            if not os.path.exists(log_folder):
                os.mkdir(log_folder)
        except:
            print("[ERROR] Cannot create logging directory")

        log_file = os.path.join(log_folder, constants.EMAIL_LOG)

        logging_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        self.logger.setLevel(logging.INFO)

        time_rotating_log_file_handler = TimedRotatingFileHandler(
            log_file, when="d", interval=1, backupCount=5)
        time_rotating_log_file_handler.setFormatter(logging_formatter)

        self.logger.addHandler(time_rotating_log_file_handler)

    def send_email(self):
        t = Thread(target=self.send, args=())
        t.daemon = True
        t.start()
        return self


    def add_attachment(self, attachment=None):

        if self.attachments is None:
            self.attachments = []

        self.attachments.append(attachment)

        return self

    def prepare_email(self):

        email_message = MIMEMultipart()
        email_message["From"] = self.sender_email
        email_message["To"] = COMMASPACE.join(self.receiver_email)
        email_message["Subject"] = "Employee Temperature Check-in"
        
        return email_message


    def send(self):
        
        email_message = self.prepare_email()
        try:
            self.logger.info("Connecting to email server")
            with SMTP_SSL(host="", port=468) as smtp_server:
                smtp_server.ehlo()
                smtp_server.starttls()
                smtp_server.login(user="", password="")
                self.logger.info("Email server connected.")
                self.logger.info("Sending email")
                smtp_server.sendmail(from_addr=self.sender_email, to_addrs=COMMASPACE.join(self.receiver_email), msg=email_message.as_string())
        except Exception as e:
            self.logger.error("Error connecting to email server.")

