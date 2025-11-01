const chatBtn = document.getElementById('chat-btn');
const chatModal = document.getElementById('chat-modal');
const closeChat = document.getElementById('close-chat');
const chatBox = document.getElementById('chat-box');
const chatInput = document.getElementById('chat-input');

chatBtn.addEventListener('click', () => {
  chatModal.classList.remove('hidden');
});

closeChat.addEventListener('click', () => {
  chatModal.classList.add('hidden');
});

chatInput?.addEventListener('keypress', (e) => {
  if (e.key === 'Enter' && chatInput.value.trim()) {
    const msg = chatInput.value.trim();
    chatBox.innerHTML += `<div class='text-right mb-2'><span class='bg-tamuccTeal text-white px-2 py-1 rounded'>${msg}</span></div>`;
    chatInput.value = '';
    setTimeout(() => {
      chatBox.innerHTML += `<div class='text-left mb-2'><span class='bg-gray-200 px-2 py-1 rounded'>I'll connect you soon 😊</span></div>`;
      chatBox.scrollTop = chatBox.scrollHeight;
    }, 500);
  }
});
