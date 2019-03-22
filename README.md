# wise_reporter

임시

Scouter 사용법
Keyword를 Query로 newspaper index에서 가져오는 소스

```
from modules.scouter_handler import ScouterHandler

scouter = ScouterHandler()
query_body = sh.make_keyword_query_body("문재인 대통령")
sh.search(query_body, data_type="newspaper")
```
