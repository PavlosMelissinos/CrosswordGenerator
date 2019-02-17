from collections import namedtuple
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer as SQLInteger, String as SQLString
import sqlalchemy


ConnectionInfo = namedtuple('ConnectionInfo', ['host', 'port', 'dbname', 'user', 'password'])

default_conn_info = ConnectionInfo(host='localhost',
                                   port='5430',
                                   dbname='crossgen',
                                   user='crossgen',
                                   password='xQ755y7Gr3Bg')


# db_tables = {
#     'words': ('id serial PRIMARY KEY',
#               'lang varchar',
#               'term varchar',
#               'pos varchar',
#               'definition varchar'),
#     'lexicon': ('id serial PRIMARY KEY',
#                 'lang varchar',
#                 'word_id int FOREIGN KEY'),
#
# }


def create_engine(user, password,
                  host=default_conn_info.host,
                  port=default_conn_info.port,
                  dbname=default_conn_info.dbname,
                  dialect='postgresql',
                  driver='psycopg2',
                  **kwargs) -> sqlalchemy.engine.Engine:
    return sqlalchemy.create_engine(f'{dialect}+{driver}://{user}:{password}@{host}:{port}/{dbname}')


def setup(db: ConnectionInfo=default_conn_info):
    engine = create_engine(**db._asdict())
    Base.metadata.create_all(engine)


Base = declarative_base()


class Word(Base):
    __tablename__ = 'words'

    id = Column(SQLInteger, primary_key=True)

    lang = Column(SQLString)
    term = Column(SQLString)
    pos = Column(SQLString)
    length = Column(SQLInteger)


if __name__ == "__main__":
    setup()
