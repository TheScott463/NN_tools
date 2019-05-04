import hashlib

def hash_function(s):
    b = s.encode()
    return hashlib.sha1(b).hexdigest()

print(hash_function('hello world'))

print(hash_function('Hello world'))


class Block():
    
    def __init__(self, previous_hash, transactions):
        self.previous_hash = previous_hash
        self.transactions = transactions

    def __repr__(self):
        return self.__dict__.__repr__()
    
    def compute_hash(self):
        return hash_function(self.previous_hash + ('%s' % self.transactions))
    

b0 = Block(hash_function(''), [])
blockchain = [b0]

blockchain[0]
print("output: {'previous_hash': 'da39a3ee5e6b4b0d3255bfef95601890afd80709', 'transactions': []}")

# ========== CLIENT 1 ==========
if (hash_function('') == b0.previous_hash):
    new_transactions = ['Alice gives 20$ to Bob.', 'Bob gives 30$ to Charlie.']
    b1 = Block(b0.compute_hash(), new_transactions)
    blockchain.append(b1)

blockchain[1]
print("output: {'previous_hash': '2a323a4cc7bcc0ae781538da4f1162a79bb62113', 'transactions': ['Alice gives 20$ to Bob.', 'Bob gives 30$ to Charlie.']}")

# ========== CLIENT 2 ==========
if (b0.compute_hash() == b1.previous_hash):
    new_transactions = ['Charlie gives 10$ to Alice.', 'David gives 5$ to Bob.']
    b2 = Block(b1.compute_hash(), new_transactions)
    blockchain.append(b2)

blockchain[2]
print("output: {'previous_hash': '60ddc9fbadf21abc622c5b542b675856324b35e2',"
      "'transactions': ['Charlie gives 10$ to Alice.', 'David gives 5$ to Bob.']}")

# ========== CLIENT 3 ==========
if (b1.compute_hash() == b2.previous_hash):
    b3 = Block(b2.compute_hash(), ['Alice gives 100$ to David.'])
    blockchain.append(b3)
    
for i in range(len(blockchain)):
    print('Block %d: %s' % (i, blockchain[i].compute_hash()))


print("output verify:"
"    Block 0: 2a323a4cc7bcc0ae781538da4f1162a79bb62113"
 "   Block 1: 60ddc9fbadf21abc622c5b542b675856324b35e2"
  "  Block 2: 939bdf9d4274462538b38158162a20b7175fd3f5"
   " Block 3: 0589681d0ecb7998ac1a4b36a898453b35db0891")


def check_integrity():
    
    for i in range(1,len(blockchain)):
        h1 = blockchain[i-1].compute_hash()
        h2 = blockchain[i].previous_hash
        print('Block %d: %s => %s' % (i-1, h1, 'VALID' if (h1==h2) else 'WRONG'))
        
check_integrity()

blockchain[1].transactions[1] = 'Bob gives 30000$ to Eve.'
blockchain[1]

check_integrity()

new_hash = blockchain[1].compute_hash()
blockchain[2].previous_hash = new_hash

check_integrity()

transactions = ['Alice gives 20$ to Bob.',
                'Bob gives 30$ to Charlie.',
                'Charlie gives 10$ to Alice.',
                'David gives 5$ to Bob.',
                'Alice gives 100$ to David.']

def hash_tree(trx):
    if len(trx) == 1:
        return hash_function(trx[0])
    elif len(trx) == 2:
        return hash_function(trx[0] + trx[1])
    else:
        return hash_function(hash_tree(trx[:2]) + hash_tree(trx[2:len(trx)]))

root_hash = hash_tree(transactions)
root_hash

class Block():
    
    def __init__(self, previous_hash, transactions):
        self.previous_hash = previous_hash
        self.root_hash = hash_tree(transactions)
        self.transactions = transactions

    def __repr__(self):
        return self.__dict__.__repr__()
    
    def compute_hash(self):
        return hash_function(self.previous_hash + self.root_hash)
    
b4 = Block(b3.compute_hash(), ['Bob gives 50$ to Alice.'])
b4

base_string = 'Hello, World!'
nonce = -1
h = ''

while h[:3] != '000':
    nonce = nonce + 1
    s = base_string + str(nonce)
    h = hash_function(s)
    
print('%s => %s' % (s, h))


def mint(challenge, work):
    nonce = -1
    h = ''
    
    while h[:work] != '0'*work:
        nonce = nonce + 1
        h = hash_function(challenge + str(nonce))
    
    return nonce

mint('Hello, World!', 3)

def evaluate(challenge, work, nonce):
    h = hash_function(challenge + str(nonce))
    return (h[:work] == '0'*work)

evaluate('Hello, World!', 3, 898)

class Block():
    
    def __init__(self, previous_hash, transactions, nonce):
        self.previous_hash = previous_hash
        self.root_hash = hash_tree(transactions)
        self.nonce = nonce
        self.transactions = transactions

    def __repr__(self):
        return self.__dict__.__repr__()
    
    def compute_hash(self):
        return hash_function(self.previous_hash + self.root_hash + str(self.nonce))


trx = ['Bob gives 50$ to Alice.']
root_hash = hash_tree(trx)
prev_hash = b3.compute_hash()
work_amount = 3

nonce = mint(prev_hash + root_hash, work_amount)
b4 = Block(prev_hash, trx, nonce)
b4

evaluate(b4.previous_hash + b4.root_hash, work_amount, b4.nonce)


