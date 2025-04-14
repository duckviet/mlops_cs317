#!/bin/bash

# Lấy commit message từ file tạm (Git truyền commit message vào file này)
COMMIT_MSG_FILE=$1
COMMIT_MSG=$(cat "$COMMIT_MSG_FILE")

# Định nghĩa các loại commit hợp lệ
VALID_COMMIT_TYPES="feature|fixbug|setup|release|fix|refactor|doc"

# Kiểm tra cú pháp commit message
# Cú pháp: <loại commit>(<tên công việc>): <comment công việc>
# Hoặc: <loại commit>: <comment công việc> (nếu không có tên công việc)
if ! echo "$COMMIT_MSG" | grep -E "^($VALID_COMMIT_TYPES)(\([a-z0-9-]+\))?: [a-z0-9].*" > /dev/null; then
    echo "ERROR: Commit message không đúng cú pháp!"
    echo "Cú pháp hợp lệ: <loại commit>(<tên công việc>): <comment công việc>"
    echo "Ví dụ: feature(create-model): create model.py file"
    echo "Loại commit hợp lệ: feature, fixbug, setup, release, fix, refactor, doc"
    echo "Tên công việc (nếu có): chỉ chứa chữ thường, số, và dấu - (e.g., create-model)"
    echo "Commit message không được viết hoa."
    echo "Commit message hiện tại: $COMMIT_MSG"
    exit 1
fi

# Kiểm tra xem commit message có viết hoa không
if echo "$COMMIT_MSG" | grep -E "[A-Z]" > /dev/null; then
    echo "ERROR: Commit message không được chứa chữ hoa!"
    echo "Commit message hiện tại: $COMMIT_MSG"
    exit 1
fi

# Nếu tất cả kiểm tra đều qua, cho phép commit
exit 0
