# Матриця ролей, систем і процесних об'єктів

- **Ідентифікатор документа:** D4
- **Призначення:** однозначне визначення контекстної компоненти `S_ctx`
- **Область:** усі атомарні вузли контрольного графа

Матриця є нормативним джерелом основної ролі та інформаційної системи для
кожного вузла. Ці атрибути використовуються для обчислення контекстної
однорідності. Бізнес-об'єкт зберігається для простежуваності, але не входить до
поточної формули `S_ctx`.

| Вузол | Нормалізована назва | Роль | Система | Бізнес-об'єкт |
|---|---|---|---|---|
| v1 | Receive user request | Service Desk | ITSM | Incident |
| v2 | Register incident | Service Desk | ITSM | Incident |
| v3 | Determine incident priority | Service Desk | ITSM | Incident |
| v4 | Classify incident | Service Desk | ITSM | Incident |
| v5 | Determine failure type | Service Desk | ITSM | Incident |
| v6 | Check account status | Identity Analyst | IAM | Account |
| v7 | Check authentication policy | Identity Analyst | IAM | Access Policy |
| v8 | Check external identity provider availability | Identity Analyst | External IdP | Identity Provider |
| v9 | Restore user access | Identity Analyst | IAM | Account |
| v10 | Check service availability | Platform Engineer | Monitoring | Service |
| v11 | Check application status | Platform Engineer | Monitoring | Application |
| v12 | Check service dependencies | Platform Engineer | Monitoring | Dependency |
| v13 | Restore software platform | Platform Engineer | Orchestrator | Deployment |
| v14 | Verify recovery result | Service Desk | ITSM | Incident |
| v15 | Determine incident outcome | Service Desk | ITSM | Incident |
| v16 | Escalate unresolved incident | Service Desk | ITSM | Incident |
| v17 | Notify user of result | Service Desk | ITSM | Notification |
| v18 | Close incident | Service Desk | ITSM | Incident |

Англомовні нормалізовані назви залишено навмисно: саме вони були входом для
обчислення текстової компоненти. Заміна їх перекладом змінює TF-IDF-вектори та
значення `S_txt`.
