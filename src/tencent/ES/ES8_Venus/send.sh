function sendMessage {
curl "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=90d4978d-25dc-4c2a-9d16-094a0c25e74e" \
   -H 'Content-Type: application/json' \
   -d '
   {
        "msgtype": "text",
        "text": {
            "content": "'$1'",
            "mentioned_list":["'$2'"]
        }
   }'
}

sendMessage $1 $2
