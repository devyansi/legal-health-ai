self.addEventListener('install', function(event) {
    console.log('✅ Service Worker installed');
});

self.addEventListener('fetch', function(event) {
    // You can add cache logic here later
});
