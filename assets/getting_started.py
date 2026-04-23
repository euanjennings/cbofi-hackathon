import os
import psycopg2

class connect_to_database():

    def __init__(self):
        self.jupyterhub_user = os.getenv('JUPYTERHUB_USER', 'unknown_user')
        try:
            connection = psycopg2.connect(user=os.environ["DB_USER"],
                                          host=os.environ["DB_HOST"],
                                          password=os.environ["DB_PASSWORD"],
                                          port="5432",
                                          database=os.environ["DB_DATABASE"])
            self.cursor = connection.cursor()
            self.description = self.cursor.description

        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL", error)
            return error

    def execute(self, query):
        user_comment = f" -- @@{self.jupyterhub_user}@@"
        query_with_comment = query + user_comment
        self.cursor.execute(query_with_comment)
        self.description = self.cursor.description

    def fetchall(self):
        return self.cursor.fetchall()
