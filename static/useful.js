function printName(){
    
    let question=document.getElementById('question').value;
    console.log("ajax 실행");
    let data= {"question":question};
    $.ajax({
        url:'/answer', //request 보낼 서버의 경로
        type:'post', // 메소드(get, post, put 등)
        data:JSON.stringify(data), //보낼 데이터
        contentType: "application/JSON; charset=utf-8",
        dataType:"text",
        success: function(data) {
            //서버로부터 정상적으로 응답이 왔을 때 실행
            console.log("통신성공");
            console.log("data: "+data);
            console.log(result[1]);
            document.getElementById("result").innerText=data.split('*')[0];
            document.getElementById("real_result").innerText=data.split('*')[1];
        },
        error: function(err) {
            //서버로부터 응답이 정상적으로 처리되지 못햇을 때 실행
            document.getElementById("result").innerText="도착실패";
        }
    });
    
    
    
    
    
}
document.getElementById('button').onclick=printName;