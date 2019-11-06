import re
import socket
from time import sleep
import json

class Etri_client():
    def __init__(self, host_name, port_number, silence):
        self.host_name = host_name
        self.port_number = port_number
        self.silence = silence

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error as err:
            print("Socket creation was failed with err : {}".format(err))

    # connect to server
    def connect(self):
        try:
            self.socket.connect((self.host_name, self.port_number))
            if not self.silence:
                print("Success to connect to server")
        except socket.error as err:
            print("Fail to connect to server with err : {}".format(err))

    def send(self, msg):
        try:
            self.socket.sendall(bytes(msg, 'utf-8'))
            self.socket.shutdown(socket.SHUT_WR)  # must exist
            if not self.silence:
                print("Success to send msg to server")
        except IOError as err:
            print("I/O exception: {}".format(err))
            print("while sending msg: {}".format(msg))
            exit(err)

    def recv(self, weight=0):
        try:
            data_list = []
            while True:
                # data = self.socket.recv(65536)
                data = self.recv_all(weight)
                if not data:
                    break
                recv_msg = data.decode('utf-8').strip()
                data_list.append(recv_msg)
                # print(recv_msg)
            if not self.silence:
                print("Success to receive msg from server")
            return ''.join(data_list)
        except IOError as err:
            print("I/O exception: {}".format(err))
            print("while receving msg")
            return None
        except UnicodeDecodeError:
            print("UnicodeDecoderError")
            print(data)
            print(len(data))

    def recv_all(self, weight=0):
        buf_size = 4096
        data = b""
        while True:
            part = self.socket.recv(buf_size)
            data = b"".join([data, part])
            # print(len(part), data)
            # 서버 과부화? 때문에 너무 빨리 던지면 결과를 잘 못 얻어 옴
            sleep(0.001 * (10 ** weight))
            if len(part) < buf_size:
                break
        return data

    def close_connection(self):
        try:
            self.socket.close()
            if not self.silence:
                print("Success to close")
        except socket.error as err:
            print("Failed to close socket with error {}".format(err))




class Etri_nlp():
    def __init__(self, host_addr="155.230.90.217", port_number=5004, silence=True):
        # etri nlp server address
        self.host_addr = host_addr
        # port number
        self.port_number = port_number
        self.client = Etri_client(self.host_addr, self.port_number, silence)

    def get_parsed_json(self, sentence, error=0):
        # error : 너무 빠른 통신으로 인해 데이터를 다 받지 못하고 decode하는 걸 방지하기 위해
        #         sleep에 weight를 줌

        # connect to server
        self.client.connect()
        # send msg
        self.client.send(sentence)
        # recv result
        parsed_sentence = self.client.recv(error)
        self.client.close_connection()
        # print(parsed_sentence)
        return json.loads(parsed_sentence)

    def make_morp_sentence(self, json_dict):
        morp_element_list = []
        for sentence in json_dict['sentence']:
            # get morp
            morp_elements = [morp_info['lemma'] for morp_info in sentence['morp']]
            morp_element_list.extend(morp_elements)
            # save to file
        return ' '.join(morp_element_list)

    # 17.12.02
    def get_noun_word(self, json_dict):
        noun_tag = {"NNG": 0, "NNP": 0, "NNB": 0, "NP": 0, "NR": 0}
        morp_element_list = []
        for sentence in json_dict['sentence']:
            # get morp
            morp_elements = [morp_info['lemma'] for morp_info in sentence['morp'] if morp_info['type'] in noun_tag]
            morp_element_list.append(morp_elements)
            # save to file
        return morp_element_list


    
def preProcess(str):
    # replace HTML tag
    str_list = str.split('\n')

    new_str = ''
    for elem in str_list:
        elem = elem.strip()
        if elem.find('=') != -1 :
            continue
        if elem[-2:] != '다.' :
            continue
        new_str += (elem + ' ')

    new_str = removeSpecialchar(new_str)

    return new_str

def removeSpecialchar(content):
    str = content
    # replace HTML tag
    str = str.replace('&nbsp;', ' ')
    str = str.replace('&lt;', '<')
    str = str.replace('&gt;', '>')
    str = str.replace('&amp;', '&')
    str = str.replace('&quot;', '\"')

    new_str = str
    pat_regex = '''[_\t\\^#‘’′\'\\'′`|·‥♥♡☆★○●◎■□▲Δ▽▼◁◀▶◇◆△©ⓒ▷※♤♠♧♣⊙◈♨☏☎☜☞【】↓→↑①-⑮「」㈜]'''

    new_str = re.sub(pat_regex, ' ', new_str)

    new_str = re.sub('\[(.*?)\]', '', new_str)
    new_str = new_str.replace('…', ' ')
    
    new_str = new_str.replace('  ', ' ')

    # remove e-mail pattern in contents
    email_regex = '[a-zA-Z0-9+-_.]+@[a-zA-z0-9-]+\.[a-zA-Z0-9-.]*'
    new_str = re.sub(email_regex, '', new_str)
    email_regex = '[a-zA-Z0-9+-_.]+@|@+[a-zA-Z0-9+-_.]*'
    new_str = re.sub(email_regex, '', new_str)

    new_str = new_str.strip()
    return new_str
'''
sample1 = "지난해까지 전남의 민간 식품제조회사에서 산업기능요원으로 군 대체복무를 한 ㄱ씨는 자신의 군 생활을

etri_nlp = Etri_nlp()
json_result = etri_nlp.get_parsed_json(sample1)
#print(str(json_result))
print(etri_nlp.make_morp_sentence(json_result))

'''