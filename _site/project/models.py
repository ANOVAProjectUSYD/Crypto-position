from app import db

class Bitcoin(db.Model):
    date = db.Column(db.Date, primary_key=True)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Float)
    marketcap = db.Column(db.Float)

    def __repr__(self):
        return '<Bitcoin {}>'.format(self.date)

