 
----

Make sure to have following env variables before starting the application.

openai.api.key=[your openai api key];
google.api.key=[your google api key];
anthropic.api.key=[your claude api key];


Run below url :
http://localhost:8080/rag/compare?query=what%20action%20is%20required%20if%20letter%20batch%20job%20is%20failed?

pass a query parameter to disable rag:
http://localhost:8080/rag/compare?ragEnabled=false&query=what%20action%20is%20required%20if%20letter%20batch%20job%20is%20failed?


----
