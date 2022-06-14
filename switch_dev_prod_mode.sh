if [[ $1 == 'production' ]]
then
    rm .env
    touch .env
    echo "ENV=production" >> .env
    echo "Switched to production environment"
else
    rm .env
    touch .env
    echo "ENV=development" >> .env
    echo "Switched to development environment"
fi
